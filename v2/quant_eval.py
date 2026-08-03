"""Measure what quantization actually costs — before and after, on metrics with headroom.

WHY NOT JUST RE-RUN THE GATES

Every deterministic gate is at 1.000 across 297 draws. A saturated metric cannot measure
a regression: 4-bit could lose real warmth, depth and coherence and still score a perfect
1.000 on "invents no citation". Re-running the gates before and after would cost hours and
prove nothing. They are kept as a floor — the hard guarantees must not break — but they are
not the instrument for this question.

THE THREE MEASUREMENTS THAT DO HAVE HEADROOM

1. PERPLEXITY on held-out `chosen` replies. The standard quantization metric, and the most
   sensitive single number available without a judge. It reads the model's own probability
   mass over text it should find likely. A few tenths of a percent is noise; several
   percent means the weights genuinely moved.

2. GREEDY DIVERGENCE. Both models answer the same 99 probes at temperature 0, which makes
   generation deterministic — so any difference is caused by the weights, not by sampling.
   (This is why `--temperature` had to be plumbed through generate_reply first; it was
   accepted and silently dropped, so every past "temperature 0" comparison ran at 1.0.)
   Reported as exact-match rate, token-level similarity, length ratio, and whether the
   same passages got cited.

3. THE GATES, as a floor. Not to detect subtle damage, but to catch gross breakage.

MoE MATTERS HERE. This model activates ~3.8B of 25.8B params per token, so a given token
depends on a handful of experts. Quantization error that lands unevenly across experts has
far less redundancy to hide in than it would in a dense model. That is the specific risk,
and divergence on a per-probe basis is what would expose it.

    # one model at a time — 52 GB or 13 GB, never both
    python -m v2.quant_eval replies --out v2/data/quant_bf16.json
    python -m v2.quant_eval replies --model v2/data/gemma4-v2-4bit --out v2/data/quant_4bit.json
    python -m v2.quant_eval ppl     --out v2/data/ppl_bf16.json
    python -m v2.quant_eval ppl     --model v2/data/gemma4-v2-4bit --out v2/data/ppl_4bit.json
    python -m v2.quant_eval compare --a v2/data/quant_bf16.json --b v2/data/quant_4bit.json \
                                    --ppl-a v2/data/ppl_bf16.json --ppl-b v2/data/ppl_4bit.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
from pathlib import Path

_CITE = re.compile(r"\[P\d+\]")


def _set_backend_env(model: str | None) -> None:
    """Select the gemma backend BEFORE api.config is imported — it reads env at import."""
    if model:
        os.environ["SATSANG_GEMMA_MODEL"] = str(Path(model).resolve())
    os.environ["SATSANG_GEN_BACKEND"] = "gemma"
    os.environ["SATSANG_UTILITY_BACKEND"] = "gemma"


# --------------------------------------------------------------------------- replies
def stage_replies(model: str | None, out: str, n: int, limit_tokens: int) -> None:
    """Answer every probe at temperature 0. Resumable per probe."""
    _set_backend_env(model)
    if limit_tokens:
        os.environ["SATSANG_GEMMA_MAX_NEW_TOKENS"] = str(limit_tokens)

    from api.pipeline import generate_reply, prepare
    from eval.probes import PROBES

    outp = Path(out)
    done: dict[str, dict] = {}
    if outp.exists():
        done = {r["message"]: r for r in json.loads(outp.read_text())["replies"]}
        print(f"resuming: {len(done)} probes already answered")

    rows = list(done.values())
    probes = PROBES[:n]
    for i, p in enumerate(probes, 1):
        msg = p["problem"]
        if msg in done:
            continue
        plan, passages = prepare(msg)
        reply = ""
        for item in generate_reply(msg, plan, passages, temperature=0.0):
            if isinstance(item, tuple) and item and item[0] == "__done__":
                reply = item[1][0]
        rows.append({"message": msg, "gate": p["gate"], "mode": plan.get("mode", "counseling"),
                     "reply": reply, "citations": sorted(set(_CITE.findall(reply))),
                     "passages": [getattr(x, "id", "") for x in passages]})
        outp.write_text(json.dumps({"model": model or "bf16+adapter", "replies": rows},
                                   ensure_ascii=False, indent=1))
        print(f"[{i:>3}/{len(probes)}] {plan.get('mode','?'):<13} {len(reply):>5} chars  {msg[:52]}")
    print(f"wrote {out} ({len(rows)} replies)")


# --------------------------------------------------------------------------- perplexity
def stage_ppl(model: str | None, out: str, n: int, pairs: str) -> None:
    """Completion-only perplexity over held-out `chosen` replies.

    Scores ONLY the reply tokens, with the prompt masked — the same objective SFT used.
    Anything else measures the model on tokens it was trained to ignore."""
    _set_backend_env(model)
    import torch

    from v2 import train_config as C
    from v2.schema import read_jsonl

    # THE REAL held-out split — train_config._split with the same seed the SFT run used,
    # so this is text the model was never trained on, framed exactly as training framed it.
    # Taking "the last n lines" instead would have silently included trained-on rows.
    _tr, ev = C._split([{"problem": p.problem, "context": p.context, "chosen": p.chosen}
                        for p in read_jsonl(pairs)])
    rows = ev[:n]
    print(f"scoring {len(rows)} held-out pairs (of {len(ev)} in the eval split) "
          f"on {model or 'bf16+adapter'}")

    # Reuse the SERVING model rather than loading a second copy. api.generate._gemma is
    # lru_cached, so under `measure` this is the exact instance that just answered the
    # probes — one 52 GB load, not two.
    #
    # This is the bug that killed the first run: `measure` called stage_replies (which
    # loads via _gemma) and then stage_ppl (which called load_base again), putting two
    # 52 GB models on an 80 GB card. It OOMed after every reply had been generated —
    # 44.35 GiB requested with 30 GiB free. The replies survived on disk; the load did not.
    from api.generate import _gemma
    mdl, tok = _gemma()
    mdl.eval()

    tot_nll, tot_tok = 0.0, 0
    for i, r in enumerate(rows, 1):
        # COMPLETION-ONLY LOSS — the prompt is masked, exactly as SFT trained it
        # (completion_only_loss=True). This is not a refinement, it is the difference
        # between a valid number and a meaningless one: 71.7% of each sequence is prompt,
        # and the model was explicitly trained NOT to predict those tokens. Scoring them
        # gave perplexity ~5200 (near-nonsense) and made 4-bit look 21% BETTER than bf16 —
        # an impossible result that was pure out-of-distribution noise on masked tokens.
        prompt = C.render_prompt(tok, r["problem"], r["context"])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(r["chosen"].strip(), add_special_tokens=False)["input_ids"]
        if not c_ids:
            continue
        # Keep the completion whole; drop the FRONT of the prompt if the pair is too long,
        # so truncation never eats the only tokens being scored.
        budget = 2048 - len(c_ids)
        if budget < 1:
            c_ids = c_ids[:2047]
            budget = 1
        p_ids = p_ids[-budget:]

        ids = torch.tensor([p_ids + c_ids], device=mdl.device)
        labels = ids.clone()
        labels[:, :len(p_ids)] = -100                     # mask the prompt
        with torch.no_grad():
            loss = mdl(input_ids=ids, attention_mask=torch.ones_like(ids),
                       labels=labels).loss.float().item()
        # HF shifts labels internally, so exactly len(c_ids)-1 positions carry loss.
        ntok = max(len(c_ids) - 1, 1)
        tot_nll += loss * ntok
        tot_tok += ntok
        if i % 25 == 0 or i == len(rows):
            cur = tot_nll / max(tot_tok, 1)
            print(f"  [{i:>4}/{len(rows)}] mean nll {cur:.4f}  ppl {pow(2.718281828, cur):.3f}")

    nll = tot_nll / max(tot_tok, 1)
    Path(out).write_text(json.dumps(
        {"model": model or "bf16+adapter", "n_pairs": len(rows), "tokens": tot_tok,
         "mean_nll": nll, "perplexity": pow(2.718281828, nll)}, indent=1))
    print(f"wrote {out}: nll {nll:.4f}  ppl {pow(2.718281828, nll):.3f}")


# --------------------------------------------------------------------------- compare
def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def stage_compare(a: str, b: str, ppl_a: str | None, ppl_b: str | None, out: str | None) -> None:
    A = {r["message"]: r for r in json.loads(Path(a).read_text())["replies"]}
    B = {r["message"]: r for r in json.loads(Path(b).read_text())["replies"]}
    shared = [m for m in A if m in B]
    print(f"comparing {len(shared)} probes answered by BOTH models "
          f"(A={json.loads(Path(a).read_text())['model']}, "
          f"B={json.loads(Path(b).read_text())['model']})\n")

    rows = []
    for m in shared:
        x, y = A[m], B[m]
        rows.append({
            "message": m, "mode": x["mode"],
            "identical": x["reply"] == y["reply"],
            "similarity": _sim(x["reply"], y["reply"]),
            "len_a": len(x["reply"]), "len_b": len(y["reply"]),
            "same_mode": x["mode"] == y["mode"],
            "same_citations": x["citations"] == y["citations"],
        })

    ident = sum(r["identical"] for r in rows)
    sim = sum(r["similarity"] for r in rows) / max(len(rows), 1)
    same_mode = sum(r["same_mode"] for r in rows)
    same_cit = sum(r["same_citations"] for r in rows)
    la = sum(r["len_a"] for r in rows) / max(len(rows), 1)
    lb = sum(r["len_b"] for r in rows) / max(len(rows), 1)

    print("GREEDY DIVERGENCE (temperature 0 — any difference is the weights)")
    print(f"  byte-identical replies      {ident}/{len(rows)}")
    print(f"  mean token similarity       {sim:.4f}   (1.000 = no drift at all)")
    print(f"  same routing decision       {same_mode}/{len(rows)}")
    print(f"  same citations cited        {same_cit}/{len(rows)}")
    print(f"  mean length  A {la:7.0f} chars   B {lb:7.0f} chars   "
          f"({(lb - la) / max(la, 1) * 100:+.1f}%)")

    by_mode: dict[str, list] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    print("\n  by mode:")
    for mode in sorted(by_mode):
        v = by_mode[mode]
        print(f"    {mode:<14} n={len(v):>3}  similarity {sum(x['similarity'] for x in v)/len(v):.4f}"
              f"  identical {sum(x['identical'] for x in v)}/{len(v)}")

    print("\n  most-diverged probes:")
    for r in sorted(rows, key=lambda r: r["similarity"])[:8]:
        print(f"    {r['similarity']:.3f}  [{r['mode']:<13}] {r['message'][:58]}")

    verdict = {}
    if ppl_a and ppl_b and Path(ppl_a).exists() and Path(ppl_b).exists():
        pa = json.loads(Path(ppl_a).read_text())
        pb = json.loads(Path(ppl_b).read_text())
        d = (pb["perplexity"] - pa["perplexity"]) / max(pa["perplexity"], 1e-9) * 100
        print("\nPERPLEXITY on held-out chosen replies")
        print(f"  A {pa['perplexity']:.4f}   B {pb['perplexity']:.4f}   ({d:+.2f}%)")
        print("  guide: <1% negligible · 1-3% modest · >5% real damage, drop to --bits 8")
        verdict["ppl_delta_pct"] = d
    else:
        print("\nPERPLEXITY: not supplied (run the `ppl` stage for both models)")

    verdict.update({"identical": ident, "n": len(rows), "similarity": sim,
                    "same_mode": same_mode, "same_citations": same_cit,
                    "len_delta_pct": (lb - la) / max(la, 1) * 100})
    if out:
        Path(out).write_text(json.dumps({"summary": verdict, "rows": rows}, indent=1,
                                        ensure_ascii=False))
        print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["replies", "ppl", "measure", "compare"],
                    help="'measure' = replies + ppl in ONE process, so the 52 GB model is "
                         "loaded once instead of twice")
    ap.add_argument("--model", default=None,
                    help="standalone model dir (merged/quantized). Omit for bf16 base+adapter.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=99)
    ap.add_argument("--max-new-tokens", type=int, default=0, help="0 = leave the served default")
    ap.add_argument("--pairs", default="v2/data/pairs.jsonl")
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--ppl-a", default=None)
    ap.add_argument("--ppl-b", default=None)
    a = ap.parse_args()

    if a.stage == "replies":
        stage_replies(a.model, a.out or "v2/data/quant_replies.json", a.n, a.max_new_tokens)
    elif a.stage == "ppl":
        stage_ppl(a.model, a.out or "v2/data/quant_ppl.json", a.n, a.pairs)
    elif a.stage == "measure":
        # Both measurements behind one model load. --out is the replies file; the
        # perplexity file is derived from it so the two always travel together.
        rep = a.out or "v2/data/quant_replies.json"
        ppl = rep.replace(".json", "_ppl.json")
        stage_replies(a.model, rep, a.n, a.max_new_tokens)
        stage_ppl(a.model, ppl, a.n, a.pairs)
    else:
        stage_compare(a.a, a.b, a.ppl_a, a.ppl_b, a.out)


if __name__ == "__main__":
    main()
