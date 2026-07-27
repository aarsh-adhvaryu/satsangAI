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


def _recall(idx, queries: list[str], allowed_traditions, allowed_text_types=None,
            k: int = config.CANDIDATE_K) -> list[Passage]:
    """Union of top candidates across queries, best score per id, above the cosine floor."""
    best: dict[str, tuple[dict, float]] = {}
    for q in queries:
        qv = embed_query(q)
        for row, score in idx.search(qv, allowed_traditions=allowed_traditions, k=k,
                                     allowed_text_types=allowed_text_types):
            cur = best.get(row["id"])
            if cur is None or score > cur[1]:
                best[row["id"]] = (row, score)
    return [Passage.from_row(r, s) for r, s in sorted(best.values(), key=lambda rs: -rs[1])
            if s >= config.MIN_SCORE][:k]


def retrieve(queries: list[str], mode: str = "counseling", top_k: int = config.TOP_K,
             rerank_query: str | None = None,
             prefer_text_types: tuple[str, ...] | None = None) -> list[Passage]:
    """Passages for the generator, best first.

    `prefer_text_types` is for requests that are ABOUT scripture rather than about a
    problem ("give me a shloka on focus"). Semantic similarity alone answers those badly:
    the enriched prose was written to explain feelings, so it beats eleven words of
    Sanskrit on every emotional query, and the person asking for a verse gets an essay.
    Rather than filter the whole search — which would throw away the context that makes a
    verse land — the verse rows are recalled SEPARATELY and pinned in front.
    """
    idx = vector_store()
    allowed = None if mode == "shastrarth" else config.COUNSELING_TRADITIONS
    rq = rerank_query or queries[0]

    candidates = _recall(idx, queries, allowed)
    if config.RERANK and candidates:
        from .rerank import rerank
        ranked = rerank(rq, candidates, top_k)
    else:
        ranked = candidates[:top_k]

    if not prefer_text_types:
        return ranked

    # A separate, restricted recall. Reranked among themselves so the pinned verse is the
    # most apt one, not merely the highest cosine.
    verses = _recall(idx, queries, allowed, allowed_text_types=prefer_text_types)
    if not verses:
        return ranked                      # nothing scriptural on this topic: say so honestly
    if config.RERANK:
        from .rerank import rerank
        verses = rerank(rq, verses, max(1, top_k // 2))
    else:
        verses = verses[: max(1, top_k // 2)]

    # Among this handful of near-equally apt verses, put a translated one first. Measured
    # over the core: `verse` rows are 63.6% translated but `poetry` and `saying` are 0%,
    # and reranking scores the ENGLISH contextual_explanation — so an untranslated
    # Gujarati kirtan can outrank a Gita shloka on an English query and then render with
    # its translation layer missing entirely. This reorders only inside the shortlist, so
    # relevance is never traded away for a verse that does not fit.
    verses.sort(key=lambda p: not str(p.translation or "").strip())

    pinned_ids = {p.id for p in verses}
    return (verses + [p for p in ranked if p.id not in pinned_ids])[:top_k]
