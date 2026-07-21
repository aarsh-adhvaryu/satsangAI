"""Side-by-side generation: SFT adapter vs DPO adapter, on HELD-OUT problems.

Answers the question the DPO metrics raised — accuracies saturated at 1.0 and
rewards/chosen went negative, which suggests DPO learned to detect the injected
boilerplate rather than to generate better replies, and may have degraded the
good responses too. This compares them empirically.

Loads the 52 GB base once and swaps LoRA adapters, so it costs one model load.

    python -m v2.compare_adapters --n 4
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="held-out problems to compare")
    ap.add_argument("--data", default=None)
    ap.add_argument("--sft", default=None)
    ap.add_argument("--dpo", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=250)
    a = ap.parse_args()

    import torch
    from peft import PeftModel

    from api.retrieve_types import Passage
    from api.verify import verify
    from v2 import train_config as C
    from v2.schema import read_jsonl

    data = a.data or str(C.PAIRS)
    sft, dpo = a.sft or str(C.SFT_OUT), a.dpo or str(C.DPO_OUT)

    C.tune_runtime()
    base, tok = C.load_base("bf16")
    base.config.use_cache = True                     # generation, not training

    model = PeftModel.from_pretrained(base, sft, adapter_name="sft")
    model.load_adapter(dpo, adapter_name="dpo")
    model.eval()

    # held-out split — same seed/fraction as training, so these were never trained on
    pairs = read_jsonl(data)
    _, ev = C._split([{"i": i} for i in range(len(pairs))])
    held = [pairs[e["i"]] for e in ev][:a.n]
    print(f"comparing on {len(held)} held-out problems\n" + "=" * 78)

    def gen(adapter: str, prompt: str) -> str:
        model.set_adapter(adapter)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    for k, p in enumerate(held, 1):
        prompt = C.render_prompt(tok, p.problem, p.context)
        # one grounding passage per pair -> [P1]; verify checks tags map to it
        psg = [Passage(id=p.grounding_ids[0], citation="", source="", tradition="",
                       score=1.0, original="", translation="",
                       contextual_explanation="", when_this_helps="", core_principle="")]
        print(f"\n### {k}. PROBLEM: {p.problem[:150]}")
        for tag in ("sft", "dpo"):
            txt = gen(tag, prompt)
            v = verify(txt, psg)
            print(f"\n--- {tag.upper()}  ({len(txt)} chars, "
                  f"unverified_refs={len(v.get('unverified_refs', []))}) ---")
            print(txt[:700] + ("…" if len(txt) > 700 else ""))
        print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
