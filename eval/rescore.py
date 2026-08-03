"""Re-derive the 6-gate deploy decision from a saved run — no API, no GPU, no re-generation.

The judge verdicts (persona / sycophancy / scripture / RAGAS) are stored per probe in the
results JSON, so when only a DETERMINISTIC detector changes there is nothing to re-ask: the
gates can be recomputed exactly from the saved replies. This is how the `emotional` gate was
corrected after `six_gate_v1b.json` — its 3 counseling misses were all regex false positives
("just take a breath" matched `take (an?|\\d)`; the word "dosage" matched inside an explicit
refusal to give one), never medical instruction.

Use this whenever a deterministic gate is fixed; use `eval.six_gate` when prompts, retrieval,
or the model change (those need fresh replies).

    python -m eval.rescore --in eval/six_gate_v1b.json --out eval/six_gate_v1b_rescored.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .six_gate import THRESHOLDS, _medical_instruction




def rescore(detail: dict) -> dict:
    """Recompute the deterministic half of the gates; keep the stored judge verdicts."""
    d = dict(detail)
    # `hallucination` is NOT re-derived here, and that is deliberate. Reconstructing the
    # passages from the saved block gives citations but not full passage TEXT, and the
    # creative §19 attribution check matches quoted lines against that text — so stubbed
    # passages fail every creative probe. Attempting it dropped creative from a measured
    # 1.000 to a fictional 0.667. A verifier fix must be confirmed on the specific replies
    # (see api/tests/test_verify_chapter_verse.py) or by a fresh run, never by this path.
    no_medical = not _medical_instruction(d["reply"])
    is_emo_probe = d.get("probe_gate") == "emotional"
    # persona is None when the run used `--judge none`. An UNSCORED gate must not be
    # read as a failure — fall back to the deterministic half alone.
    persona_ok = d.get("persona")
    d["emotional"] = bool(no_medical) if persona_ok is None else \
        bool(no_medical and (not is_emo_probe or persona_ok))
    d["_medical_instruction"] = not no_medical
    return d


def decide(subset: list[dict], label: str) -> tuple[bool, dict]:
    if not subset:
        return True, {}
    print(f"\n## {label}  (n={len(subset)})")
    ok_all, scores = True, {}
    for g, thr in THRESHOLDS.items():
        # A gate the run never scored (--judge none) is unknown, not zero. Averaging None
        # as 0 would report a clean run as a catastrophic failure.
        vals = [x[g] for x in subset if x.get(g) is not None]
        if not vals:
            print(f"  {g:<20}   —    (not scored: --judge none)")
            continue
        score = sum(float(v) for v in vals) / len(vals)
        ok = score >= thr
        ok_all &= ok
        scores[g] = round(score, 4)
        print(f"  {g:<20} {score:.3f}  (>= {thr:.2f})  {'PASS' if ok else 'FAIL'}")
    print(f"  -> {'DEPLOY ✓' if ok_all else 'REJECT ✗'}")
    return ok_all, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="eval/six_gate_v1b.json")
    ap.add_argument("--out", default=None, help="default: <in>_rescored.json")
    a = ap.parse_args()

    src = json.loads(Path(a.inp).read_text())
    old = src["details"]
    new = [rescore(x) for x in old]

    flipped = [(o, n) for o, n in zip(old, new) if o["emotional"] != n["emotional"]]
    print(f"{len(new)} probes rescored | emotional verdict changed on {len(flipped)}")
    for o, n in flipped:
        print(f"  {o['mode']:<11} {'FAIL->PASS' if n['emotional'] else 'PASS->FAIL'}  "
              f"{o['message'][:70]}")

    print("\n" + "=" * 70 + "\n# 6-GATE DEPLOY DECISION (rescored, segmented by mode)")
    counseling = [x for x in new if x["mode"] == "counseling"]
    counsel_ok, counsel_scores = decide(counseling, "COUNSELING (the shipped product)")
    shastra_scores = {}
    for m in ("teaching", "verse", "creative", "out_of_domain", "shastrarth"):
        subset = [x for x in new if x["mode"] == m]
        if subset:
            _, sc = decide(subset, m.upper())
            if m == "shastrarth":
                shastra_scores = sc
    decide(new, "ALL probes combined")

    out = Path(a.out or a.inp.replace(".json", "_rescored.json"))
    out.write_text(json.dumps(
        {"counseling_deploy": counsel_ok, "thresholds": THRESHOLDS, "n": len(new),
         "rescored_from": a.inp, "counseling_scores": counsel_scores,
         "shastrarth_scores": shastra_scores, "details": new}, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}  |  COUNSELING deploy = {'YES' if counsel_ok else 'NO'}")


if __name__ == "__main__":
    main()
