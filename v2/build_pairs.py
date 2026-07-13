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
import random

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


CHOSEN = {"placeholder": chosen_placeholder}


def build(n: int, chosen_kind: str, seed: int, out: str) -> None:
    if chosen_kind not in CHOSEN:
        raise SystemExit(
            f"--chosen {chosen_kind} not wired yet (offline/GPU step). "
            f"Available now: {sorted(CHOSEN)}. See module docstring.")
    gen = CHOSEN[chosen_kind]
    rng = random.Random(seed)
    pairs: list[PreferencePair] = []
    for problem, passages, row_id in seeds.sample_seeds(str(config.INDEX_PATH), n, seed):
        chosen = gen(problem, passages, rng)
        rejected, flaw = reject.make_rejected(chosen, passages, rng)
        pairs.append(PreferencePair(
            problem=problem,
            context=context_from_passages(passages),
            chosen=chosen,
            rejected=rejected,
            flaw=flaw,
            pair_source="scripture_derived",
            grounding_ids=[row_id],
            meta={"chosen_kind": chosen_kind, "tradition": passages[0].tradition,
                  "source": passages[0].source},
        ))
    path = write_jsonl(pairs, out)
    # quick distribution report
    from collections import Counter
    flaws = Counter(p.flaw for p in pairs)
    trads = Counter(p.meta["tradition"] for p in pairs)
    print(f"wrote {len(pairs)} pairs -> {path}")
    print("  flaw mix     :", dict(flaws))
    print("  tradition mix:", dict(trads))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--chosen", default="placeholder", help="placeholder | claude | gemma")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="v2/data/pairs_sample.jsonl")
    a = ap.parse_args()
    build(a.n, a.chosen, a.seed, a.out)


if __name__ == "__main__":
    main()
