"""Canonical enrichment prompt. Mirrors satsangai/pipeline/enrich.py so the KB
driver's `local` backend and this app layer produce identical field semantics.

Produces the 4 retrieval-target fields per row:
  contextual_explanation, when_this_helps, core_principle, (gujarati_explanation).
"""
from __future__ import annotations

SYSTEM = (
    "You are a scholar of Hindu and Swaminarayan scripture. You write retrieval "
    "and application metadata for a problem-first counseling AI. Be faithful to the "
    "text; never invent doctrine, names, or citations. Output STRICT JSON only."
)

PROMPT = """Given a verse or teaching, write metadata that helps a counseling AI \
retrieve and apply it to real human problems. Be faithful to the text; do not invent \
doctrine.

Citation: {citation}
Tradition: {tradition}
Original ({lang}): {original}
Transliteration: {transliteration}
Translation: {translation}

Return STRICT JSON with keys:
- "contextual_explanation": 2-3 sentences in plain modern English: what this teaching \
means for someone's life.
- "when_this_helps": one sentence naming the life situations/emotions it applies to.
- "core_principle": a 3-6 word summary phrase.{gujarati_key}
JSON only, no preamble."""

GUJARATI_KEY = '\n- "gujarati_explanation": the contextual_explanation in natural Gujarati.'

FIELDS = ("contextual_explanation", "when_this_helps", "core_principle",
          "gujarati_explanation")


def _clip(v, n=1500) -> str:
    return (str(v) if v is not None else "")[:n] or "(none)"


def build_prompt(row: dict) -> str:
    is_sw = row.get("tradition") == "swaminarayan"
    return PROMPT.format(
        citation=_clip(row.get("citation"), 200),
        tradition=row.get("tradition", ""),
        lang=row.get("lang_original", "?"),
        original=_clip(row.get("original")),
        transliteration=_clip(row.get("transliteration"), 800),
        translation=_clip(row.get("translation")),
        gujarati_key=GUJARATI_KEY if is_sw else "",
    )
