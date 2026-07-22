"""Regenerate the training pairs with the two biggest quality fixes (CPU + Claude API).

Diagnosis from this session:
  1. TRAIN/SERVE MISMATCH — every training context had 1 passage, but V1 serves
     config.TOP_K (6, reranked). The model never learned to pick among passages,
     ignore distractors, or cite [P2]-[P6].
  2. UNNATURAL PROMPTS — problems were regex-mangled `when_this_helps` strings
     ("I feel X, struggles with Y, seeks Z"), not how real people write.

This rebuilds each pair as production actually looks:
  seed problem -> naturalize (Haiku) -> retrieve TOP_K reranked passages (the real
  serving path) -> grounded `chosen` (Sonnet, verifier-gated) -> rule negative.

CPU + API only (retrieval runs on CPU like V1; set SATSANG_EMBED_DEVICE=cuda to use a
GPU if one is free). Concurrent + resumable: appends, skips seeds already done.

    python -m v2.regenerate_pairs --in v2/data/pairs.jsonl --out v2/data/pairs_v2.jsonl --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _done_seed_ids(out: Path) -> set[str]:
    if not out.exists():
        return set()
    ids = set()
    for line in out.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            sid = json.loads(line).get("meta", {}).get("seed_id")
            if sid:
                ids.add(sid)
        except json.JSONDecodeError:
            continue
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="v2/data/pairs.jsonl",
                    help="existing pairs (source of seed problems + grounding anchor)")
    ap.add_argument("--out", default="v2/data/pairs_v2.jsonl")
    ap.add_argument("--workers", type=int, default=8, help="concurrent Claude calls")
    ap.add_argument("--model", default=None, help="chosen-generation model (default sonnet)")
    ap.add_argument("--nat-model", default=None, help="naturalizer model (default haiku)")
    ap.add_argument("--no-naturalize", action="store_true",
                    help="keep the original problem text; only fix the passage count")
    a = ap.parse_args()

    import anthropic

    from api.retrieve import retrieve
    from v2 import chosen as chosen_mod, reject
    from v2.schema import PreferencePair, context_from_passages, read_jsonl

    inp, outp = Path(a.inp), Path(a.out)
    src = read_jsonl(inp)
    # seed = (original problem, stable anchor id). Anchor on the first grounding id.
    seeds = [(p.problem, (p.grounding_ids or [f"row{i}"])[0]) for i, p in enumerate(src)]
    done = _done_seed_ids(outp)
    todo = [(prob, sid) for prob, sid in seeds if sid not in done]
    print(f"{len(seeds)} seeds | {len(done)} done | {len(todo)} to regenerate | {a.workers} workers")
    if not todo:
        print("nothing to do — pairs_v2 is complete.")
        return

    client = anthropic.Anthropic()
    model = a.model or chosen_mod.GEN_MODEL
    nat_model = a.nat_model or chosen_mod.NATURALIZE_MODEL
    retr_lock = threading.Lock()          # BGE model forward is not concurrency-safe
    file_lock = threading.Lock()
    outp.parent.mkdir(parents=True, exist_ok=True)

    def work(item):
        problem, seed_id = item
        natural = problem if a.no_naturalize else \
            chosen_mod.naturalize_problem(problem, client, nat_model)
        with retr_lock:
            passages = retrieve([natural], mode="counseling")     # TOP_K, reranked
        if not passages:
            return None
        reply = chosen_mod.generate_claude(natural, passages, client, model)
        if not reply:                                             # failed verifier gate
            return None
        rng = random.Random(hash(seed_id) & 0xffffffff)
        rejected, flaw = reject.make_rejected(reply, passages, rng)
        return PreferencePair(
            problem=natural, context=context_from_passages(passages),
            chosen=reply, rejected=rejected, flaw=flaw,
            pair_source="scripture_derived_v2",
            grounding_ids=[p.id for p in passages],
            meta={"seed_id": seed_id, "negatives": "rule",
                  "n_passages": len(passages),
                  "tradition": passages[0].tradition, "source": passages[0].source})

    kept = dropped = 0
    with outp.open("a") as fh, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = [ex.submit(work, it) for it in todo]
        for fut in as_completed(futures):
            pair = fut.result()
            if pair is None:
                dropped += 1
                continue
            with file_lock:
                fh.write(json.dumps(pair.to_json(), ensure_ascii=False) + "\n")
                fh.flush()
            kept += 1
            if (kept + dropped) % 100 == 0:
                print(f"  ...{kept} kept / {dropped} dropped", flush=True)

    # report the fix landed
    import statistics as st
    npass = [p.meta.get("n_passages", 1) for p in read_jsonl(outp)]
    print(f"done: {kept} pairs -> {outp} ({dropped} dropped). "
          f"passages/context: median {st.median(npass)} (was 1). Re-run to resume.")


if __name__ == "__main__":
    main()
