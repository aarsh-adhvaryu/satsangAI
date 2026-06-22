"""Retrieve node — deterministic, no LLM. Embeds the planner's search queries with
BGE-M3, recalls candidates from the enriched counseling core with a tradition filter,
then (optionally) reranks them with a cross-encoder for precision.

Counseling mode keeps to the home tradition + shared Hindu and never mixes the acharya
schools; Shastrarth mode opens the full breadth.
"""
from __future__ import annotations

from . import config
from .embed import embed_query
from .retrieve_types import Passage
from .store import vector_store

__all__ = ["Passage", "retrieve"]


def retrieve(queries: list[str], mode: str = "counseling", top_k: int = config.TOP_K,
             rerank_query: str | None = None) -> list[Passage]:
    idx = vector_store()
    allowed = None if mode == "shastrarth" else config.COUNSELING_TRADITIONS

    # 1. vector recall: union of top candidates across queries, best score per id
    best: dict[str, tuple[dict, float]] = {}
    for q in queries:
        qv = embed_query(q)
        for row, score in idx.search(qv, allowed_traditions=allowed, k=config.CANDIDATE_K):
            cur = best.get(row["id"])
            if cur is None or score > cur[1]:
                best[row["id"]] = (row, score)
    candidates = [Passage.from_row(r, s) for r, s in
                  sorted(best.values(), key=lambda rs: -rs[1])
                  if s >= config.MIN_SCORE][:config.CANDIDATE_K]

    # 2. rerank (precision) or fall back to vector order
    if config.RERANK and candidates:
        from .rerank import rerank
        return rerank(rerank_query or queries[0], candidates, top_k)
    return candidates[:top_k]
