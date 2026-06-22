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
# scripture-name + number patterns that should have come via a [P#] tag
_LOOSE_REF = re.compile(
    r"\b(Bhagavad\s+Gita|Gita|Vachanamrut|Swamini\s*Vato|Shikshapatri|Satsang\s+Diksha|"
    r"[A-Z][a-z]+\s+Upanishad|Yoga\s+Sutra[s]?)\b[^.\n]{0,20}\d", re.I)


def verify(text: str, passages: list[Passage]) -> dict:
    idx = vector_store()
    used = sorted({int(m) for m in _TAG.findall(text)})
    cited = []
    for n in used:
        if 1 <= n <= len(passages):
            p = passages[n - 1]
            cited.append({"tag": f"[P{n}]", "citation": p.citation, "source": p.source,
                          "id": p.id, "exists": idx.citation_exists(p.citation)})

    # loose references not tagged → flag (strip tags first so [P#] context is removed)
    untagged = _TAG.sub(" ", text)
    flagged = sorted({m.group(0).strip() for m in _LOOSE_REF.finditer(untagged)})

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
