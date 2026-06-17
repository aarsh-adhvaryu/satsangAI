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
    plus the explicitly listed shared_hindu sources."""
    m = manifest or load_manifest()
    core = m["core"]
    trad = set(core.get("traditions") or [])
    srcs = set(core.get("sources") or [])
    return df["tradition"].isin(trad) | df["source"].isin(srcs)


def select_core(df: pd.DataFrame, manifest: dict | None = None) -> pd.DataFrame:
    return df[core_mask(df, manifest)].copy()
