"""Select a diverse ~N-row sample of the counseling core to become the QLoRA
gold seed (the rows Claude will enrich offline, then we tune Gemma 4 26B MoE on).

Goal: maximize coverage of the dimensions the tuned model must generalize over —
tradition, source, text_type, original language, and translation-present vs not —
WITHOUT drowning in the few huge sw_lit biographies.

Allocation: per-source budget ~ proportional to sqrt(row_count) (dampens big
sources, lifts rare high-value texts like Shikshapatri / Yoga Sutras), clamped to
[floor, cap]. Within each source, draw stratified across (text_type, lang_original)
so verse/prose/discourse and gu/sa/en are all represented. Deterministic (seeded).

Usage:
    python -m enrichment.sample_gold_seed \
        --corpus ../satsangai/data/parquet/corpus.parquet \
        --out enrichment/data/gold_seed_sample.parquet --target 1500
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .core_filter import select_core

# Columns the enricher needs to read (keep the sample lean — no embeddings).
KEEP = ["id", "source", "tradition", "text_type", "citation", "ref",
        "lang_original", "original", "transliteration", "translation",
        "word_meanings", "commentaries"]


def _alloc(counts: pd.Series, target: int, floor: int, cap: int) -> dict[str, int]:
    """sqrt-weighted budget per source, clamped to [floor, cap], scaled to ~target."""
    w = counts.pow(0.5)
    raw = (w / w.sum()) * target
    budget = raw.round().clip(lower=floor, upper=cap).astype(int)
    # never ask for more rows than a source has
    budget = budget.combine(counts, min)
    return budget.to_dict()


def _draw(group: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Stratified draw of n rows across (text_type, lang_original) strata."""
    if n >= len(group):
        return group
    strata = list(group.groupby(["text_type", "lang_original"], dropna=False))
    rng.shuffle(strata)
    picks: list[int] = []
    # round-robin one row per stratum until budget filled
    pools = [s.index.to_list() for _, s in strata]
    for p in pools:
        rng.shuffle(p)
    while len(picks) < n and any(pools):
        for p in pools:
            if p:
                picks.append(p.pop())
                if len(picks) >= n:
                    break
    return group.loc[picks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../satsangai/data/parquet/corpus.parquet")
    ap.add_argument("--out", default="enrichment/data/gold_seed_sample.parquet")
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--floor", type=int, default=4, help="min rows per source")
    ap.add_argument("--cap", type=int, default=60, help="max rows per source")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    df = pd.read_parquet(args.corpus, columns=KEEP)
    core = select_core(df)
    print(f"core rows: {len(core)} across {core.source.nunique()} sources")

    counts = core.source.value_counts()
    budget = _alloc(counts, args.target, args.floor, args.cap)
    rng = np.random.default_rng(args.seed)

    parts = [_draw(core[core.source == s], n, rng) for s, n in budget.items() if n > 0]
    sample = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out, index=False)

    print(f"\nwrote {len(sample)} rows -> {out}")
    print("\n=== coverage ===")
    print("tradition:\n", sample.tradition.value_counts().to_string())
    print("text_type:\n", sample.text_type.value_counts().to_string())
    print("lang_original:\n", sample.lang_original.value_counts().to_string())
    has_tr = sample.translation.fillna("").str.strip().ne("")
    print(f"has_translation: {int(has_tr.sum())} / {len(sample)}")
    print(f"sources covered: {sample.source.nunique()} / {core.source.nunique()}")


if __name__ == "__main__":
    main()
