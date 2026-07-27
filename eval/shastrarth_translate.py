"""EXPERIMENT: does translating the Sanskrit grounding fix shastrarth's failing gates?

Shastrarth fails hallucination (0.78) and scripture_accuracy (0.83). Diagnosis from the
index: all 8,173 acharya-school rows carry `original` (raw Devanagari OCR, avg 2,185 chars)
and NOTHING else — translation, transliteration, word_meanings and contextual_explanation
are 0% populated, and `commentaries` is an empty 2-char placeholder. So `_passages_block`
falls through to `p.translation or p.original` and hands the model untranslated Sanskrit
(truncated to 600 chars), then asks for precise scholarly claims. It fills the gaps.

This measures the fix BEFORE paying for it. Two arms, SAME retrieved passages:
  A (baseline) — grounding exactly as served today
  B (translated) — each passage's Sanskrit translated to English offline, put in
                   `translation` so the block renders readable grounding

If B lifts hallucination/scripture, the ~4h GPU enrichment of all 8,173 school rows is
justified (offline, same A1-A4 pipeline). If it doesn't, the problem is elsewhere and the
GPU session is saved. Translation is OFFLINE EVAL ONLY — it never enters the runtime path,
so the deterministic citation verifier stays the only thing standing behind a [P#].

    python -m eval.shastrarth_translate --limit 6      # cheap first look
    python -m eval.shastrarth_translate                # all stored shastrarth probes
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

_CACHE = Path("eval/.translation_cache.json")

_TRANSLATE_SYS = (
    "You are a Sanskrit scholar. Translate the following passage from a classical Vedantic "
    "commentary (bhashya) into clear, literal English. Preserve technical terms with a short "
    "gloss on first use (e.g. 'brahman (the absolute)'). The text is OCR output and may be "
    "imperfect — translate what is legible and write [illegible] for what is not. Do NOT add "
    "interpretation, doctrine, or content that is not in the passage. Output ONLY the "
    "translation.")


def _load_cache() -> dict:
    if _CACHE.exists():
        try:
            return json.loads(_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _translate(client, cache: dict, pid: str, sanskrit: str, model: str) -> str:
    if pid in cache:
        return cache[pid]
    resp = client.messages.create(
        model=model, max_tokens=1500, system=_TRANSLATE_SYS,
        messages=[{"role": "user", "content": sanskrit[:2400]}])
    out = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    cache[pid] = out
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="eval/six_gate_v1b.json",
                    help="run whose shastrarth probe messages are replayed")
    ap.add_argument("--limit", type=int, default=None, help="only the first N probes (cheap)")
    ap.add_argument("--model", default="claude-sonnet-4-6", help="translation model (offline)")
    ap.add_argument("--out", default="eval/shastrarth_translate.json")
    a = ap.parse_args()

    from api.generate import stream_reply
    from api.retrieve import retrieve
    from api.understand import understand
    from .six_gate import THRESHOLDS, _score_one

    import anthropic
    client = anthropic.Anthropic()
    cache = _load_cache()

    src = json.loads(Path(a.baseline).read_text())
    msgs = [d["message"] for d in src["details"] if d["mode"] == "shastrarth"]
    if a.limit:
        msgs = msgs[:a.limit]
    print(f"replaying {len(msgs)} shastrarth probes | translation model {a.model}\n" + "=" * 78)

    rows = []
    for i, msg in enumerate(msgs, 1):
        plan = understand(msg)
        mode = plan.get("mode", "shastrarth")
        passages = retrieve(plan["search_queries"], mode=mode,
                            rerank_query=plan.get("problem_summary") or msg)

        # --- arm A: exactly what is served today -------------------------------- #
        reply_a = "".join(stream_reply(msg, plan, passages))
        score_a = _score_one(msg, passages, reply_a, "scripture_accuracy", mode)

        # --- arm B: same passages, Sanskrit translated offline ------------------- #
        translated = []
        for p in passages:
            if p.translation.strip():
                translated.append(p)                      # already readable
                continue
            en = _translate(client, cache, p.id, p.original, a.model)
            translated.append(dataclasses.replace(p, translation=en))
        reply_b = "".join(stream_reply(msg, plan, translated))
        score_b = _score_one(msg, translated, reply_b, "scripture_accuracy", mode)

        n_tr = sum(1 for p, q in zip(passages, translated) if not p.translation.strip()
                   and q.translation.strip())
        print(f"[{i:>2}/{len(msgs)}] translated {n_tr}/{len(passages)} passages | "
              f"hall {int(score_a['hallucination'])}->{int(score_b['hallucination'])} "
              f"scrip {int(score_a['scripture_accuracy'])}->{int(score_b['scripture_accuracy'])} "
              f"ragas {score_a['ragas']:.2f}->{score_b['ragas']:.2f} | {msg[:52]}")
        rows.append({"message": msg, "n_translated": n_tr,
                     "baseline": {**score_a, "reply": reply_a},
                     "translated": {**score_b, "reply": reply_b}})

    print("\n" + "=" * 78 + "\n# SHASTRARTH: raw Sanskrit grounding vs translated grounding\n")
    print(f"  {'gate':<20} {'baseline':>9} {'translated':>11} {'delta':>8}   threshold")
    verdict = {}
    for g, thr in THRESHOLDS.items():
        base = sum(float(r["baseline"][g]) for r in rows) / len(rows)
        tran = sum(float(r["translated"][g]) for r in rows) / len(rows)
        verdict[g] = {"baseline": round(base, 4), "translated": round(tran, 4),
                      "delta": round(tran - base, 4), "threshold": thr}
        flag = "  <-- fixes gate" if (tran >= thr > base) else ""
        print(f"  {g:<20} {base:>9.3f} {tran:>11.3f} {tran - base:>+8.3f}   >= {thr:.2f}{flag}")

    fixed = [g for g, v in verdict.items() if v["translated"] >= v["threshold"] > v["baseline"]]
    print(f"\n  gates flipped to PASS by translation: {fixed or 'none'}")
    print("  -> " + ("ENRICHMENT JUSTIFIED (run the ~4h GPU job on the 8,173 school rows)"
                     if fixed else
                     "translation alone is NOT the fix — investigate retrieval/alignment first"))

    Path(a.out).write_text(json.dumps(
        {"n": len(rows), "verdict": verdict, "gates_fixed": fixed, "rows": rows},
        indent=2, ensure_ascii=False))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
