"""Rebuild the DPO pairs with ON-POLICY negatives — the fix for the saturated run.

The first DPO hit rewards/accuracies = 1.0 at every eval and loss ~0 by step 75,
because `rejected` was the chosen text plus one of a handful of canned strings
(~475 repeats each across 4,750 pairs). The model learned "contains this boilerplate
= bad" instead of "sycophancy/hallucination = bad".

Fix: sample `rejected` from the **SFT model itself**. Its own output is a near-miss —
same register, same grounding, genuinely worse in the ways that matter — so there is
no surface pattern to shortcut. `chosen` stays the Claude gold, unchanged (no new
Claude spend). A configurable slice keeps the deterministic rule-based negatives,
because those encode hard rules (fabricated citation) worth training explicitly.

GPU (samples from the SFT adapter). Resumable: appends, skips done grounding-ids.

    python -m v2.onpolicy_negatives --sft v2/data/gemma4-v2-sft-lora \\
        --in v2/data/pairs.jsonl --out v2/data/pairs_onpolicy.jsonl --batch 8
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None, help="pairs.jsonl with the Claude `chosen`")
    ap.add_argument("--out", default="v2/data/pairs_onpolicy.jsonl")
    ap.add_argument("--sft", default=None, help="SFT adapter to sample negatives from")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=260)
    ap.add_argument("--temperature", type=float, default=1.0, help="higher = weaker negatives")
    ap.add_argument("--rule-frac", type=float, default=0.2,
                    help="fraction keeping the deterministic flaw-injected rejected")
    a = ap.parse_args()

    import torch
    from peft import PeftModel

    from v2 import reject, train_config as C
    from v2.build_pairs import _done_ids
    from v2.schema import PreferencePair, read_jsonl
    from api.retrieve_types import Passage

    inp = Path(a.inp or str(C.PAIRS))
    outp = Path(a.out)
    sft = a.sft or str(C.SFT_OUT)

    pairs = read_jsonl(inp)
    done = _done_ids(outp)
    todo = [p for p in pairs if not (set(p.grounding_ids) & done)]
    print(f"{len(pairs)} pairs | {len(done)} already done | {len(todo)} to generate")
    if not todo:
        print("nothing to do — on-policy pair set is complete.")
        return

    C.tune_runtime()
    base, tok = C.load_base("bf16")
    base.config.use_cache = True
    model = PeftModel.from_pretrained(base, sft)
    model.eval()
    tok.padding_side = "left"                      # decoder-only batched generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rng = random.Random(0)
    outp.parent.mkdir(parents=True, exist_ok=True)
    kept = ruled = 0
    with outp.open("a") as fh:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            prompts = [C.render_prompt(tok, p.problem, p.context) for p in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                                     do_sample=True, temperature=a.temperature, top_p=0.95,
                                     pad_token_id=tok.pad_token_id)
            gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

            for p, samp in zip(chunk, gen):
                samp = samp.strip()
                use_rule = rng.random() < a.rule_frac or len(samp) < 40 or samp == p.chosen.strip()
                if use_rule:
                    # keep an explicit rule negative (also the fallback if the sample is degenerate)
                    psg = [Passage(id=p.grounding_ids[0], citation="", source="",
                                   tradition=p.meta.get("tradition", ""), score=1.0,
                                   original="", translation="", contextual_explanation="",
                                   when_this_helps="", core_principle="")]
                    rejected, flaw = reject.make_rejected(p.chosen, psg, rng)
                    ruled += 1
                else:
                    rejected, flaw = samp, "on_policy_sft"
                np = PreferencePair(
                    problem=p.problem, context=p.context, chosen=p.chosen,
                    rejected=rejected, flaw=flaw, pair_source=p.pair_source,
                    grounding_ids=p.grounding_ids,
                    meta={**p.meta, "negatives": "on_policy" if not use_rule else "rule"})
                fh.write(json.dumps(np.to_json(), ensure_ascii=False) + "\n")
                kept += 1
            fh.flush()
            if kept % 200 < a.batch:
                print(f"  ...{kept}/{len(todo)} ({ruled} rule-based)", flush=True)

    print(f"done: {kept} pairs -> {outp} ({ruled} rule-based, {kept - ruled} on-policy). "
          f"Re-run to resume.")


if __name__ == "__main__":
    main()
