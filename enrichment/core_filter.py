"""Resolve the counseling-core row set from config/counseling_core.yaml.

Single source of truth for "which rows are the counseling core" — used by the
gold-seed sampler, the enrichment driver, and (later) V1 retrieval filtering.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

CONFIG = Path(__file__).resolve().parent.parent / "config" / "counseling_core.yaml"


def load_manifest(path: Path | str = CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def core_mask(df: pd.DataFrame, manifest: dict | None = None) -> pd.Series:
    """Boolean mask selecting the Tier-1 counseling core: the home-tradition rows
    plus the explicitly listed shared_hindu sources, minus non-scripture front/back
    matter (by RULE — see enrichment/non_scripture.py) and any `excluded_ids`.

    `excluded_ids` is for manual one-offs only. It cannot carry the load on its own:
    ids change on every re-chunk, and in 2026-07 a re-chunk silently unbound 8 of 14
    exclusions, putting a conference title page and a list of abbreviations back into
    the served counseling index. The rule survives re-chunking; the id list does not.
    """
    m = manifest or load_manifest()
    core = m["core"]
    trad = set(core.get("traditions") or [])
    srcs = set(core.get("sources") or [])
    mask = df["tradition"].isin(trad) | df["source"].isin(srcs)

    if m.get("exclude_non_scripture", True):
        # Needs the enrichment layer to read. Callers that run BEFORE enrichment (the
        # gold-seed sampler, the enrichment driver itself) legitimately have no such
        # columns and simply skip the rule — there is nothing yet to judge.
        if {"contextual_explanation", "when_this_helps"} <= set(df.columns):
            from .non_scripture import mask as non_scripture_mask
            mask &= ~non_scripture_mask(df)

    excluded = set(m.get("excluded_ids") or [])
    if excluded:
        mask &= ~df["id"].isin(excluded)
    return mask


def select_core(df: pd.DataFrame, manifest: dict | None = None) -> pd.DataFrame:
    return df[core_mask(df, manifest)].copy()
