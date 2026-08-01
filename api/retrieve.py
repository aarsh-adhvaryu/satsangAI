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
            k: int = config.CANDIDATE_K, allowed_sources=None) -> list[Passage]:
    """Union of top candidates across queries, best score per id, above the cosine floor."""
    best: dict[str, tuple[dict, float]] = {}
    for q in queries:
        qv = embed_query(q)
        for row, score in idx.search(qv, allowed_traditions=allowed_traditions, k=k,
                                     allowed_text_types=allowed_text_types,
                                     allowed_sources=allowed_sources):
            cur = best.get(row["id"])
            if cur is None or score > cur[1]:
                best[row["id"]] = (row, score)
    return [Passage.from_row(r, s) for r, s in sorted(best.values(), key=lambda rs: -rs[1])
            if s >= config.MIN_SCORE][:k]


def retrieve(queries: list[str], mode: str = "counseling", top_k: int = config.TOP_K,
             rerank_query: str | None = None,
             prefer_text_types: tuple[str, ...] | None = None,
             prefer_sources: tuple[str, ...] | None = None) -> list[Passage]:
    """Passages for the generator, best first.

    `prefer_text_types` is for requests that are ABOUT scripture rather than about a
    problem ("give me a shloka on focus"). Semantic similarity alone answers those badly:
    the enriched prose was written to explain feelings, so it beats eleven words of
    Sanskrit on every emotional query, and the person asking for a verse gets an essay.
    Rather than filter the whole search — which would throw away the context that makes a
    verse land — the verse rows are recalled SEPARATELY and pinned in front.

    `prefer_sources` does the same for a request that NAMES a text ("the Shikshapatri verse
    on non-violence"). Measured 2026-08-01: that exact question returned the Gita,
    Satsangijivanam, Harililamrut and a children's primer — and none of the 212 addressable
    Shikshapatri shlokas, four of which are about non-violence. The model then quoted the
    primer's wording accurately and captioned it "Shikshapatri, Verse 12". Real text,
    invented attribution — the worst failure this system can produce, and it happened
    because naming a text did nothing to retrieval. Same shape of fix as prefer_text_types:
    a separate restricted recall pinned in front, so the named text is present without
    discarding the surrounding context.
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

    if prefer_sources:
        # Query the named source by TOPIC ONLY. Leaving the book's name in the query makes
        # it match the verses that name the book rather than the ones about the subject —
        # see verse.strip_source_names for the measurement.
        from .verse import strip_source_names
        topic_qs = [strip_source_names(q) for q in queries]
        topic_rq = strip_source_names(rq)
        named = _recall(idx, topic_qs, allowed, allowed_sources=prefer_sources)
        if named:
            if config.RERANK:
                from .rerank import rerank
                named = rerank(topic_rq, named, top_k)
            # Half the window at most: the named text must be present, but the passages
            # that make it land are what turn a quotation into an answer.
            named = named[:max(1, top_k // 2)]
            keep = {p.id for p in named}
            ranked = (named + [p for p in ranked if p.id not in keep])[:top_k]

    if not prefer_text_types:
        return ranked

    # A separate, restricted recall. Reranked among themselves so the pinned verse is the
    # most apt one, not merely the highest cosine.
    from .verse import is_colophon
    verses = [p for p in _recall(idx, queries, allowed, allowed_text_types=prefer_text_types)
              if not is_colophon(p.original, p.translation)]
    if not verses:
        return ranked                      # nothing scriptural on this topic: say so honestly
    n_verse = max(1, top_k // 2)
    if config.RERANK:
        from .rerank import rerank
        # Reranked to a WIDER window than we keep, so the translated-first pass below has
        # something to promote. Rerank straight to n_verse and the preference is inert:
        # it can only reorder three rows that are already chosen.
        verses = rerank(rq, verses, top_k)

    # Put a translated verse first. Measured over the core: `verse` rows are 63.6%
    # translated but `poetry` and `saying` are 0%, and reranking scores the ENGLISH
    # contextual_explanation — so an untranslated Gujarati kirtan outranks a Gita shloka on
    # an English query and then renders with its translation layer missing entirely. The
    # sort is stable and the window is small, so relevance ordering survives within each
    # group and nothing outside the top few can be promoted.
    verses.sort(key=lambda p: not str(p.translation or "").strip())
    verses = verses[:n_verse]

    pinned_ids = {p.id for p in verses}
    return (verses + [p for p in ranked if p.id not in pinned_ids])[:top_k]
