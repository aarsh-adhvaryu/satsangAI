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


def _norm_ref(s: str) -> str:
    """Canonical form for comparing a written reference to a KB citation."""
    s = re.sub(r"\s+", " ", str(s or "").lower()).strip()
    s = re.sub(r"\s*[.:\-]\s*", ".", s)
    return re.sub(r"\bbhagavad gita\b|\bgita\b", "gita", s)


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
    untagged = _TAG.sub(" ", text)
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
