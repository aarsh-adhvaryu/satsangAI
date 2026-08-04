"""Verify node — deterministic, no LLM. Confirms every citation in the reply is real.

By design the generator cites with [P#] tags, so each maps to a retrieved passage whose
citation exists in the KB (double-checked against the index). Any scripture-looking
reference the model wrote WITHOUT a tag is flagged as unverified (possible hallucination).
"""
from __future__ import annotations

import re

from .store import vector_store
from .retrieve import Passage

_TAG = re.compile(r"\[P(\d+)\]")
# Scripture-name + number references. Captures the FULL number ("2.47", not "2") so the
# reference can be matched against the passages actually in context.
_LOOSE_REF = re.compile(
    r"\b(Bhagavad\s+Gita|Gita|Vachanamrut|Swamini\s*Vato|Shikshapatri|Satsang\s+Diksha|"
    r"[A-Z][a-z]+\s+Upanishad|Yoga\s+Sutra[s]?)\b[^.\n\d]{0,20}"
    r"(\d+(?:\s*[.:\-]\s*\d+)*)", re.I)


# People — and models — write a reference two ways: "Gita 2.20" and "Bhagavad Gita,
# Chapter 2, Verse 20". _LOOSE_REF only understands the first, so the second parsed as
# "Bhagavad Gita, Chapter 2" and was flagged as UNVERIFIED even when the passage for
# 2.20 was right there in context. That is a false accusation of hallucination against a
# correct citation, and it fires in PRODUCTION, not just in the eval — the user sees a
# grounding warning on a right answer. Found 2026-08-03 when the 4-bit model phrased the
# same correct verse the long way and "failed" a gate the bf16 model passed 3/3.
_CHAPTER_VERSE = re.compile(
    r"\bchapter\s*(\d+)\s*[,;]?\s*(?:verse|shloka|sloka|sutra)\s*(\d+)", re.I)


def _expand_prose_refs(text: str) -> str:
    """Rewrite 'Chapter 2, Verse 20' as '2.20' so both spellings match the same citation."""
    return _CHAPTER_VERSE.sub(r"\1.\2", str(text or ""))


def _norm_ref(s: str) -> str:
    """Canonical form for comparing a written reference to a KB citation."""
    s = re.sub(r"\s+", " ", str(s or "").lower()).strip()
    s = s.replace(",", " ")                       # "gita, 2.20" -> "gita 2.20"
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*[.:\-]\s*", ".", s)
    s = re.sub(r"\bbhagavad gita\b|\bgita\b", "gita", s)
    # Singular/plural: the KB cites "Yoga Sutras 1.3" but "Yoga Sutra 1.3" is the more
    # natural English for ONE sutra, and prefix matching accepts neither as the other.
    # Measured 2026-08-04: the 12B wrote the singular and had two correct, in-context
    # citations flagged as hallucinations — in production a user would see a grounding
    # warning on a right answer. Fold the plural away on both sides.
    return re.sub(r"\b(sutra|upanishad|veda|purana)s\b", r"\1", s)


def verify(text: str, passages: list[Passage]) -> dict:
    idx = vector_store()
    used = sorted({int(m) for m in _TAG.findall(text)})
    cited = []
    for n in used:
        if 1 <= n <= len(passages):
            p = passages[n - 1]
            cited.append({"tag": f"[P{n}]", "citation": p.citation, "source": p.source,
                          "id": p.id, "exists": idx.citation_exists(p.citation)})

    # Loose (untagged) references. A reference is only a hallucination risk if it names
    # something that is NOT among the retrieved passages — naming a verse that IS in
    # context is accurate, not invented, and §5.2 verse mode does it in every heading.
    # Anything outside the retrieved set still flags: that is the model drawing on
    # parametric memory, which is exactly what this check exists to catch.
    in_context = {_norm_ref(p.citation) for p in passages}
    untagged = _expand_prose_refs(_TAG.sub(" ", text))
    flagged = []
    for m in _LOOSE_REF.finditer(untagged):
        written = _norm_ref(m.group(0))
        if any(written == c or c.startswith(written) or written.startswith(c)
               for c in in_context):
            continue                      # resolves to a passage we actually retrieved
        flagged.append(m.group(0).strip())
    flagged = sorted(set(flagged))

    return {
        "cited": cited,
        "unverified_refs": flagged,
        "all_ok": all(c["exists"] for c in cited) and not flagged,
    }


def render_citations(text: str, passages: list[Passage]) -> str:
    """Expand [P#] tags to human-readable citations for display."""
    def sub(m):
        n = int(m.group(1))
        return f"({passages[n-1].citation})" if 1 <= n <= len(passages) else m.group(0)
    return _TAG.sub(sub, text)
