"""Retrieve node — deterministic, no LLM. Embeds the planner's search queries with
BGE-M3, searches the enriched counseling core with a tradition filter, merges/dedups
across queries, and returns the top grounded passages.

Counseling mode keeps to the home tradition + shared Hindu and never mixes the acharya
schools; Shastrarth mode opens the full breadth.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .embed import embed_query
from .index import get_index


@dataclass
class Passage:
    id: str
    citation: str
    source: str
    tradition: str
    score: float
    original: str
    translation: str
    contextual_explanation: str
    when_this_helps: str
    core_principle: str

    @classmethod
    def from_row(cls, row: dict, score: float) -> "Passage":
        g = lambda k: str(row.get(k) or "")
        return cls(id=row["id"], citation=g("citation"), source=g("source"),
                   tradition=g("tradition"), score=score, original=g("original"),
                   translation=g("translation"),
                   contextual_explanation=g("contextual_explanation"),
                   when_this_helps=g("when_this_helps"), core_principle=g("core_principle"))


def retrieve(queries: list[str], mode: str = "counseling",
             top_k: int = config.TOP_K) -> list[Passage]:
    idx = get_index()
    allowed = None if mode == "shastrarth" else config.COUNSELING_TRADITIONS

    best: dict[str, tuple[dict, float]] = {}
    for q in queries:
        qv = embed_query(q)
        for row, score in idx.search(qv, allowed_traditions=allowed):
            cur = best.get(row["id"])
            if cur is None or score > cur[1]:
                best[row["id"]] = (row, score)

    ranked = sorted(best.values(), key=lambda rs: -rs[1])
    passages = [Passage.from_row(r, s) for r, s in ranked if s >= config.MIN_SCORE]
    return passages[:top_k]
