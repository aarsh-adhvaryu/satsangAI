"""Build the scripture-derived DPO preference set (CPU, offline-first).

Pipeline:  seeds (mine when_this_helps) -> grounding context -> chosen reply
           -> rejected (Claude-free flaw injection) -> JSONL.

The `chosen` side is pluggable so V2's Claude-free purity call isn't hard-coded:
  --chosen placeholder  (default) : grounded template — runs NOW, no model/API;
                                     use it to smoke-test the whole pipeline.
  --chosen claude                 : Claude Opus, OFFLINE gold only (bootstrap;
                                     mirrors the enrichment decision, reversible).
  --chosen gemma                  : the SFT'd V2 Gemma self-generates (strict
                                     Claude-free) — needs a GPU session.

Only `placeholder` runs without extra deps; claude/gemma are wired as TODO hooks
so the data schema, seeds, and negatives are all testable before any GPU/spend.

    python -m v2.build_pairs --n 200 --chosen placeholder --out v2/data/pairs_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from api import config
from api.retrieve_types import Passage
from v2 import reject, seeds
from v2.schema import PreferencePair, context_from_passages, write_jsonl


def chosen_placeholder(problem: str, passages: list[Passage], rng: random.Random) -> str:
    """A grounded, non-sycophantic saint reply built from the passage — no model.
    Realistic enough for the injectors to corrupt (has a [P1] tag, sw doctrine,
    depth, a gentle push-back)."""
    p = passages[0]
    meaning = (p.contextual_explanation or p.translation or "").strip().rstrip(".")
    principle = (p.core_principle or "").strip().rstrip(".")
    return (
        "I hear you, and I'm grateful you shared this with me. "
        "What you're carrying is real, and it deserves an honest look rather than a quick fix. "
        f"There is a teaching that speaks to exactly this: {meaning}. [P1] "
        "I won't pretend the hard part away — sitting with it, gently, is itself part of the path. "
        f"If there is one thing to hold onto, let it be this: {principle.lower()}. "
        "Be patient with yourself; you don't have to carry it all at once."
    )


def _make_pair(problem: str, passages: list[Passage], row_id: str,
               chosen: str, chosen_kind: str, rng: random.Random) -> PreferencePair:
    rejected, flaw = reject.make_rejected(chosen, passages, rng)
    return PreferencePair(
        problem=problem, context=context_from_passages(passages),
        chosen=chosen, rejected=rejected, flaw=flaw,
        pair_source="scripture_derived", grounding_ids=[row_id],
        meta={"chosen_kind": chosen_kind, "tradition": passages[0].tradition,
              "source": passages[0].source})


def _done_ids(out: Path) -> set[str]:
    """Resume support: grounding-ids already written to `out`. Tolerates a
    truncated final line from a hard crash mid-write (skips unparseable lines)."""
    if not out.exists():
        return set()
    ids = set()
    for line in out.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.update(json.loads(line).get("grounding_ids", []))
        except json.JSONDecodeError:
            continue                                    # partial/corrupt line — ignore
    return ids


def build_offline(n: int, chosen_kind: str, seed: int, out: str,
                  model: str, workers: int) -> None:
    """Claude-bootstrapped `chosen` (offline gold), verifier-gated, resumable, concurrent."""
    import anthropic
    from v2 import chosen as chosen_mod
    outp = Path(out)
    done = _done_ids(outp)
    all_seeds = seeds.sample_seeds(str(config.INDEX_PATH), n, seed)
    todo = [s for s in all_seeds if s[2] not in done]
    print(f"chosen={chosen_kind} model={model} | {len(all_seeds)} seeds, "
          f"{len(done)} already done, {len(todo)} to generate, {workers} workers")
    client = anthropic.Anthropic()
    outp.parent.mkdir(parents=True, exist_ok=True)

    def work(item):
        problem, passages, row_id = item
        reply = chosen_mod.generate_claude(problem, passages, client, model)
        return item, reply

    kept = dropped = 0
    # Write each pair the moment it finishes (as_completed, not map) so a crash
    # loses at most the in-flight calls, never completed-but-unyielded ones. Each
    # line is a full flush -> append-only file is always resume-safe.
    with outp.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, it) for it in todo]
        for fut in as_completed(futures):
            (problem, passages, row_id), reply = fut.result()
            if not reply:                                   # failed the verifier gate
                dropped += 1
                continue
            pair = _make_pair(problem, passages, row_id, reply, chosen_kind,
                              random.Random(hash(row_id) & 0xffffffff))
            fh.write(json.dumps(pair.to_json(), ensure_ascii=False) + "\n")
            fh.flush()
            kept += 1
            if (kept + dropped) % 50 == 0:
                print(f"  ...{kept} kept / {dropped} dropped", flush=True)
    print(f"done: {kept} pairs appended -> {outp} ({dropped} dropped by verifier gate). "
          f"Re-run the same command to resume/top-up.")


def build_placeholder(n: int, seed: int, out: str) -> None:
    """Templated grounded `chosen` — CPU, no API. For pipeline smoke tests only."""
    rng = random.Random(seed)
    pairs = [_make_pair(problem, passages, row_id,
                        chosen_placeholder(problem, passages, rng), "placeholder", rng)
             for problem, passages, row_id in seeds.sample_seeds(str(config.INDEX_PATH), n, seed)]
    path = write_jsonl(pairs, out)
    from collections import Counter
    print(f"wrote {len(pairs)} pairs -> {path}")
    print("  flaw mix     :", dict(Counter(p.flaw for p in pairs)))
    print("  tradition mix:", dict(Counter(p.meta["tradition"] for p in pairs)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--chosen", default="placeholder", help="placeholder | claude | gemma")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="v2/data/pairs_sample.jsonl")
    ap.add_argument("--model", default="claude-sonnet-4-6", help="chosen=claude generation model")
    ap.add_argument("--workers", type=int, default=8, help="concurrent Claude calls (chosen=claude)")
    a = ap.parse_args()
    if a.chosen == "placeholder":
        build_placeholder(a.n, a.seed, a.out)
    elif a.chosen == "claude":
        build_offline(a.n, a.chosen, a.seed, a.out, a.model, a.workers)
    elif a.chosen == "gemma":
        raise SystemExit("--chosen gemma (strict Claude-free self-gen) needs an SFT adapter "
                         "first — bootstrap with --chosen claude, SFT, then wire gemma self-gen.")
    else:
        raise SystemExit(f"unknown --chosen {a.chosen}")


if __name__ == "__main__":
    main()
