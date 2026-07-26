"""Verse explanation (proposal §5.2) — present a verse in full, deterministically.

Nothing here uses an LLM. Everything is lookup + rule-based transliteration, so the
layered display (original / transliteration / translation / word-by-word / meaning) is
as trustworthy as the citation verifier: the model narrates a verse it cannot alter.

Two data realities shape this module, both measured against the live index:

1. `original` is NOT all Devanagari. Across the 17,794-row core: Gujarati 8,466,
   **Latin 7,208**, Devanagari 1,476, Kannada 643. Shikshapatri and Satsang Diksha ship
   already-romanised, and the Upanishads are a mix of Kannada, Devanagari and Latin. So
   the source script is detected per verse; hardcoding Devanagari silently passes other
   scripts through unchanged (the bug that surfaced on an Atharvana Upanishad row).

2. Only some texts are verse-addressable. Gita (719) and Yoga Sutras (195) use
   'Name C.V'; Vachanamrut (273) uses 'Vachanamrut GI-1: title'. Shikshapatri, Satsang
   Diksha and the Upanishads are stored as PAGE chunks ('… p1-2'), so "Shikshapatri
   verse 12" cannot resolve — those fall back to ordinary semantic retrieval.
"""
from __future__ import annotations

import functools
import re

# Unicode block -> the sanscript scheme name used for transliteration.
_SCRIPT_RANGES = [
    ("DEVANAGARI", 0x0900, 0x097F), ("BENGALI", 0x0980, 0x09FF),
    ("GURMUKHI", 0x0A00, 0x0A7F), ("GUJARATI", 0x0A80, 0x0AFF),
    ("ORIYA", 0x0B00, 0x0B7F), ("TAMIL", 0x0B80, 0x0BFF),
    ("TELUGU", 0x0C00, 0x0C7F), ("KANNADA", 0x0C80, 0x0CFF),
    ("MALAYALAM", 0x0D00, 0x0D7F),
]


def detect_script(text: str) -> str | None:
    """Dominant Indic script of `text`, or None if it is already Latin/unknown."""
    text = str(text or "")
    best, best_n = None, 0
    for name, lo, hi in _SCRIPT_RANGES:
        n = sum(1 for c in text if lo <= ord(c) <= hi)
        if n > best_n:
            best, best_n = name, n
    latin = sum(1 for c in text if "a" <= c.lower() <= "z")
    if best_n == 0 or best_n < latin * 0.2:
        return None                       # already romanised (or no Indic content)
    return best


def to_latin(text: str) -> str:
    """Rule-based transliteration of any Indic script to IAST. Latin passes through."""
    src = detect_script(text)
    if src is None:
        return str(text or "")
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        return ""                         # dependency missing: show nothing, never guess
    return transliterate(str(text), getattr(sanscript, src), sanscript.IAST)


# --------------------------------------------------------------------------- #
#  Reference parsing + exact lookup                                            #
# --------------------------------------------------------------------------- #
_TEXT_ALIASES = {
    "bhagavad gita": "Bhagavad Gita", "bhagavad-gita": "Bhagavad Gita",
    "bhagavadgita": "Bhagavad Gita", "gita": "Bhagavad Gita", "geeta": "Bhagavad Gita",
    "bg": "Bhagavad Gita", "yoga sutra": "Yoga Sutras", "yoga sutras": "Yoga Sutras",
    "yogasutra": "Yoga Sutras", "patanjali": "Yoga Sutras",
    "vachanamrut": "Vachanamrut", "vachanamrit": "Vachanamrut",
}
# "Gita 2.47" / "BG 2:47" / "Yoga Sutras 1.2" / "Gita chapter 2 verse 47"
_NUMERIC_REF = re.compile(
    r"(?P<text>bhagavad[\s-]?gita|bhagavadgita|gita|geeta|bg|yoga\s*sutras?|yogasutra|patanjali)"
    r"[\s,]*(?:chapter\s*)?(?P<ch>\d+)\s*(?:[.:\-]|\s+verse\s+|\s+shloka\s+)\s*(?P<v>\d+)",
    re.I)
# "Vachanamrut Gadhada I-11" / "Vachanamrut GI-11" / "GII-3"
_VACHANAMRUT_REF = re.compile(
    r"(?:vachanamrut|vachanamrit)?\s*(?P<sec>[A-Z]{1,2}[IVX]{0,3})\s*[-\s]\s*(?P<n>\d+)", re.I)
_SECTION_ALIASES = {"gadhada": "G", "sarangpur": "S", "kariyani": "K", "loya": "L",
                    "panchala": "P", "vartal": "V", "amdavad": "A", "ahmedabad": "A"}


def parse_reference(message: str) -> str | None:
    """Extract a canonical citation prefix from free text, or None.

    Returns the string the KB's `citation` column starts with, e.g. 'Bhagavad Gita 2.47'.
    """
    msg = str(message or "")
    m = _NUMERIC_REF.search(msg)
    if m:
        name = _TEXT_ALIASES.get(re.sub(r"[\s-]+", " ", m.group("text").lower().strip()))
        if name:
            return f"{name} {int(m.group('ch'))}.{int(m.group('v'))}"
    low = msg.lower()
    if "vachanamrut" in low or "vachanamrit" in low or re.search(r"\bg[i]{1,3}\b", low):
        spelled = msg
        for word, code in _SECTION_ALIASES.items():          # 'Gadhada I-11' -> 'G I-11'
            spelled = re.sub(word, code, spelled, flags=re.I)
        # then close the gap the substitution leaves: 'G I-11' -> 'GI-11', 'G 1-11' -> 'GI-11'
        spelled = re.sub(r"\b([A-Z])\s+([IVX]{1,3})\s*[-\s]\s*(\d+)", r"\1\2-\3", spelled)
        spelled = re.sub(r"\b([A-Z])\s*([123])\s*[-.\s]\s*(\d+)",
                         lambda m: f"{m.group(1)}{'I' * int(m.group(2))}-{m.group(3)}", spelled)
        m = _VACHANAMRUT_REF.search(spelled)
        if m:
            return f"Vachanamrut {m.group('sec').upper()}-{int(m.group('n'))}"
    return None


@functools.lru_cache(maxsize=1)
def _citation_index() -> dict:
    """citation-prefix -> row dict, built once from the served index."""
    import pandas as pd
    from . import config
    df = pd.read_parquet(config.INDEX_PATH)
    out: dict[str, dict] = {}
    for row in df.to_dict("records"):
        cit = str(row.get("citation") or "").strip()
        if not cit:
            continue
        key = cit.split(":")[0].strip().lower()      # 'Vachanamrut GI-1: title' -> prefix
        out.setdefault(key, row)
    return out


def lookup(reference: str) -> dict | None:
    """Exact verse lookup by canonical citation. None when the text isn't addressable."""
    return _citation_index().get(str(reference).strip().lower())


def verse_view(row: dict) -> dict:
    """The layered §5.2 display for one verse. `transliteration` is stored when the KB
    has it (the KB's own romanisation convention) and generated only when it does not."""
    g = lambda k: str(row.get(k) or "").strip()
    original = g("original")
    stored = g("transliteration")
    script = detect_script(original)
    return {
        "citation": g("citation"),
        "source": g("source"),
        "id": g("id"),
        "script": script or "Latin",
        "original": original,
        "transliteration": stored or (to_latin(original) if script else ""),
        "transliteration_generated": bool(not stored and script),
        "translation": g("translation"),
        "word_meanings": g("word_meanings"),
        "meaning": g("contextual_explanation"),
        "gujarati": g("gujarati_explanation"),
    }


def render_block(view: dict) -> str:
    """Plain-text layered rendering handed to the generator as fixed context.

    The model narrates around this block; it never rewrites it. Missing layers are simply
    absent — an honest gap beats an invented transliteration or a guessed word gloss.
    """
    parts = [f"VERSE {view['citation']} ({view['source']})"]
    if view["original"]:
        parts.append(f"original ({view['script']}):\n{view['original'][:800]}")
    if view["transliteration"]:
        tag = " [generated]" if view["transliteration_generated"] else ""
        parts.append(f"transliteration{tag}:\n{view['transliteration'][:800]}")
    if view["translation"]:
        parts.append(f"translation:\n{view['translation'][:800]}")
    if view["word_meanings"]:
        parts.append(f"word-by-word:\n{view['word_meanings'][:1200]}")
    if view["meaning"]:
        parts.append(f"meaning:\n{view['meaning'][:800]}")
    return "\n\n".join(parts)
