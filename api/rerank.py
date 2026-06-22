"""Cross-encoder reranking with BGE-reranker-v2-m3 (multilingual, pairs with BGE-M3).

Vector search gives recall; the cross-encoder judges each (query, passage) pair
directly for precision. Reranks on the counseling MEANING of the passage
(contextual_explanation + when_this_helps), falling back to the verse text.

CPU works (small candidate sets); SATSANG_EMBED_DEVICE=cuda for low latency.
"""
from __future__ import annotations

import functools
import os

from . import config
from .retrieve_types import Passage


@functools.lru_cache(maxsize=1)
def _model():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL, device=config.EMBED_DEVICE, max_length=512)


def _passage_text(p: Passage) -> str:
    meaning = " ".join(x for x in (p.contextual_explanation, p.when_this_helps) if x)
    return (meaning or p.translation or p.original)[:1200]


def rerank(query: str, passages: list[Passage], top_k: int) -> list[Passage]:
    if not passages:
        return []
    scores = _model().predict([(query, _passage_text(p)) for p in passages])
    order = sorted(range(len(passages)), key=lambda i: -scores[i])
    out = []
    for i in order[:top_k]:
        passages[i].rerank_score = float(scores[i])
        out.append(passages[i])
    return out
