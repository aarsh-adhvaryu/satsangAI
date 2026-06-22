"""Shared retrieval type (kept separate so rerank.py and retrieve.py don't cycle)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Passage:
    id: str
    citation: str
    source: str
    tradition: str
    score: float                       # vector cosine
    original: str
    translation: str
    contextual_explanation: str
    when_this_helps: str
    core_principle: str
    rerank_score: float | None = None  # cross-encoder score (set if reranked)

    @classmethod
    def from_row(cls, row: dict, score: float) -> "Passage":
        g = lambda k: str(row.get(k) or "")
        return cls(id=row["id"], citation=g("citation"), source=g("source"),
                   tradition=g("tradition"), score=score, original=g("original"),
                   translation=g("translation"),
                   contextual_explanation=g("contextual_explanation"),
                   when_this_helps=g("when_this_helps"), core_principle=g("core_principle"))
