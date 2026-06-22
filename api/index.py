"""In-memory counseling-core index: the embedding matrix for retrieval + by-id and
by-citation lookups for grounding and the deterministic citation check. Loaded once.

(Postgres is the proposal's production store; for V1 the ~17.8k-row parquet in memory
is simpler and fast. The lookup API here is what a Postgres swap would back later.)
"""
from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd

from . import config

_TEXT_COLS = ["id", "source", "text_type", "tradition", "citation", "ref",
              "lang_original", "original", "transliteration", "translation",
              "contextual_explanation", "when_this_helps", "core_principle",
              "gujarati_explanation", "verified"]


def _norm_citation(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


class Index:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.meta = self.df[_TEXT_COLS]
        self.emb = np.asarray(self.df["embedding"].tolist(), dtype="float32")  # (N,1024) unit-norm
        self.by_id = {r["id"]: r for r in self.meta.to_dict("records")}
        # citation -> id, for the deterministic "does this citation exist" check
        self.by_citation = {_norm_citation(c): i for c, i in
                            zip(self.meta["citation"], self.meta["id"]) if c}

    def search(self, qvec: np.ndarray, allowed_traditions=None, k=config.CANDIDATE_K):
        """Cosine (dot, both unit-norm) search, optional tradition allowlist."""
        scores = self.emb @ qvec.astype("float32")
        if allowed_traditions is not None:
            mask = self.meta["tradition"].isin(allowed_traditions).to_numpy()
            scores = np.where(mask, scores, -1.0)
        top = np.argpartition(-scores, range(min(k, len(scores))))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.meta.iloc[i].to_dict(), float(scores[i])) for i in top]

    def citation_exists(self, citation: str) -> bool:
        return _norm_citation(citation) in self.by_citation

    def get(self, row_id: str) -> dict | None:
        return self.by_id.get(row_id)


@functools.lru_cache(maxsize=1)
def get_index() -> Index:
    return Index(pd.read_parquet(config.INDEX_PATH))
