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


# Texts addressed by a SINGLE number rather than chapter.verse. Kept separate from
# _NUMERIC_REF because that pattern requires a chapter/verse separator and so can never
# match "Satsang Diksha 12" — the citation form of two of the most-quoted Swaminarayan
# scriptures. Swamini Vato is excluded deliberately: it is cited "1/21" (prakaran/vato),
# which _SINGLE_REF would silently truncate to prakaran 1.
_SINGLE_TEXTS = {
    r"satsang\s*diksha": "Satsang Diksha",
    r"satsangdiksha": "Satsang Diksha",
    r"shikshapatri": "Shikshapatri",
    r"shikshpatri": "Shikshapatri",
}
_SINGLE_REF = re.compile(
    r"(?P<text>" + "|".join(_SINGLE_TEXTS) + r")\s*(?:shlok(?:a)?|verse|no\.?|#)?\s*"
    r"(?P<n>\d{1,3})(?!\s*[.:]\s*\d)", re.I)


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
    m = _SINGLE_REF.search(msg)
    if m:
        key = re.sub(r"\s+", " ", m.group("text").lower().strip())
        for pat, name in _SINGLE_TEXTS.items():
            if re.fullmatch(pat, key, re.I):
                return f"{name} {int(m.group('n'))}"
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


# --------------------------------------------------------------------------- #
#  Unnamed verse requests                                                      #
# --------------------------------------------------------------------------- #
# The rows a person means by "a shloka": scripture lines, kirtan poetry, and the
# aphoristic sayings (Swamini Vato). Everything else in the core — prose, commentary,
# discourse — is ABOUT scripture rather than being it, and is what "give me a shlok on
# focus" used to return: three pages of a saint's biography, because biography prose
# discusses focus at length while Gita 6.35 mentions it in eleven Sanskrit words.
VERSE_TEXT_TYPES = ("verse", "poetry", "saying")

# Chapter colophons ("oṃ tatsaditi ... thus ends the Nth chapter") are stored in the KB as
# ordinary verses: 18 of them sit in the served index as text_type='verse', numbered one
# past the end of their chapter (Bhagavad Gita 10.43, 11.56, 12.21 ...). They are enriched,
# so they rank like real verses — but their translation is the literal string "Swami
# Sivananda did not comment on this sloka". Offering one as "a shloka about devotion" is
# indefensible, so they are excluded from verse SEARCH. An exact lookup of 10.43 still
# resolves: there the stored explanation correctly says it is a closing marker.
_COLOPHON_TEXT = re.compile(r"ॐ\s*तत्सदिति|इति\s+श्रीमद्भगवद्गीता|इति\s+श्रीमद्?भगवद्गीतासु")
_NO_COMMENT = re.compile(r"did not comment on this", re.I)


def is_colophon(original: str = "", translation: str = "") -> bool:
    """True for a chapter-ending formula masquerading as a verse.

    Matched against the OPENING of the text, not anywhere in it: a colophon row *is* the
    formula, whereas a commentary page chunk may legitimately quote one mid-page (five
    Vallabhacharya Anubhashya pages do exactly that, and must not be suppressed).
    """
    return bool(_COLOPHON_TEXT.search(str(original or "")[:80])
                or _NO_COMMENT.search(str(translation or "")))

_VERSE_NOUN = (r"(?:shlokas?|shloks?|slokas?|verses?|sutras?|couplets?|dohas?|"
               r"quotations?|quotes?)")
# Asked FOR one ("give me a verse", "any shloka", "which sutra") ...
_VERSE_ASK = re.compile(
    rf"\b(?:give|share|tell|show|find|send|recite|read|need|want|looking\s+for|is\s+there|"
    rf"any|some|a|an|one|which|what)\b[^.?!]{{0,40}}\b{_VERSE_NOUN}\b", re.I)
# ... or named one and said what it should be about ("a verse about anger").
_VERSE_TOPIC = re.compile(rf"\b{_VERSE_NOUN}\b\s*(?:about|on|for|regarding|related\s+to)\b",
                          re.I)
# "what does the Gita say about duty" — a scripture request without the word 'verse'.
_SCRIPTURE_SAYS = re.compile(
    r"\b(?:gita|geeta|vachanamrut|vachanamrit|upanishads?|yoga\s*sutras?|shikshapatri|"
    r"satsang\s*diksha|swamini\s*vato|scriptures?|shastras?)\b[^.?!]{0,30}"
    r"\b(?:says?|said|say|teach(?:es)?|tells?|mentions?|states?)\b", re.I)


def wants_verse(message: str) -> bool:
    """True when someone is asking for scripture ITSELF, not for counsel about a problem.

    Deliberately separate from `parse_reference`: that one answers "which verse", this one
    answers "a verse at all". A named reference is an exact lookup; an unnamed request is
    still a search, but it must search the verse rows rather than the whole core.
    """
    msg = str(message or "")
    return bool(_VERSE_ASK.search(msg) or _VERSE_TOPIC.search(msg)
                or _SCRIPTURE_SAYS.search(msg))


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


_WBW_HEADING = re.compile(
    r"(?im)^[\s>*_#-]*\**\s*word[\s-]*by[\s-]*word\b.*$|^[\s>*_#-]*\**\s*word\s+meanings\b.*$")

# §17 morphological analysis — dhatu, case endings, compound structure — is in the
# proposal but has NO STORAGE ANYWHERE IN THE KB. There is no morphology column, and
# `word_meanings` carries flat glosses ("śhreyān | better"), never grammar. So a
# grammatical analysis is ALWAYS recalled from the model rather than retrieved — for
# every verse, including the 701 Gita rows that do have stored glosses.
#
# That is why this guard is UNCONDITIONAL where the word-by-word guard is conditional.
# Measured 2026-08-01: the 5 morphology probes scored persona 0.357 / scripture 0.500,
# the worst of any mode, on replies that read beautifully — "śhreyān is a comparative
# adjective in the nominative singular" is exactly the confident, unverified Sanskrit
# the zero-hallucination guarantee exists to catch.
_GRAMMAR_HEADING = re.compile(
    r"(?im)^[\s>*_#-]*\**\s*("
    r"(the\s+)?(sanskrit\s+)?grammar\b"
    r"|grammatical\s+(analysis|breakdown|notes?|structure)"
    r"|morpholog\w*"
    r"|(word|verbal|sanskrit)\s+roots?\b|dh[aā]tu\b|etymolog\w*"
    r"|case\s+endings?\b|declension|conjugation"
    r"|compound\s+(analysis|structure|breakdown)|sam[aā]sa\b"
    r").*$")

# A skipped section runs until one of the real verse layers resumes. Grammar headings
# are listed here too, so stripping a word-by-word block stops cleanly where the
# grammar block begins and the grammar strip can then handle it.
_NEXT_HEADING = re.compile(r"(?im)^[\s>*_#-]*\**\s*(meaning|translation|original|"
                           r"transliteration|explanation|what this means|"
                           r"grammar|grammatical|morpholog)\b")

_NO_WBW_NOTE = ("*(A word-by-word breakdown isn't recorded for this verse in my sources, "
                "so I won't invent one.)*")

_NO_GRAMMAR_NOTE = ("*(A grammatical breakdown — roots, case endings, compound structure — "
                    "isn't recorded for this verse in my sources. I won't reconstruct one "
                    "from memory, because I'd have no way to show you it was right.)*")


def claims_word_by_word(reply: str) -> bool:
    return bool(_WBW_HEADING.search(reply))


def claims_grammar(reply: str) -> bool:
    return bool(_GRAMMAR_HEADING.search(reply))


# A gloss has a SHAPE, independent of what the section is called: an emphasised Sanskrit
# token, a dash, a definition. Matching the shape rather than the heading is deliberate.
#
# Measured 2026-08-01: the heading guard worked and the model routed around it. Asked to
# break down Yoga Sutras 1.2 it wrote "*(No word-by-word gloss was supplied for this
# verse…)*" — the honest decline the guard exists to produce — and then, under the
# heading "Breaking Down the Three Key Words", supplied exactly that gloss anyway:
# "**Chitta** — the mind-stuff", "**Vṛtti** — literally 'whirlpool'". Same fabrication,
# relabelled. Only 2 of 62 verse replies tripped a heading pattern; the judge failed far
# more, because the content was there under other names.
_TOKEN_GLOSS = re.compile(
    r"(?m)^\W{0,4}[*_]{1,2}\s*"
    r"([A-Za-zāīūṛṝḷṅñṭḍṇśṣḥṃṁ][A-Za-zāīūṛṝḷṅñṭḍṇśṣḥṃṁ'’-]{1,24})"
    r"\s*[*_]{1,2}\s*[—–-]{1,2}\s+\S")


def claims_token_glosses(reply: str, min_hits: int = 2) -> bool:
    """True when the reply defines individual Sanskrit words, whatever it calls the section.

    Only meaningful when the verse has NO stored word_meanings — the Gita's 701 glossed
    rows render a legitimate gloss table of exactly this shape. Two hits are required so
    a single emphasised term inside ordinary explanation is not mistaken for a glossary.
    """
    return len(_TOKEN_GLOSS.findall(reply)) >= min_hits


def _strip_section(reply: str, heading: re.Pattern, note: str) -> str:
    """Drop every line from `heading` until a real verse layer resumes, leaving `note`.

    Fails CLOSED: if no recognised layer heading follows, the skip runs to the end of
    the reply. Removing too much is recoverable; leaving invented Sanskrit in place is
    not.
    """
    out, skipping = [], False
    for line in reply.splitlines():
        if heading.match(line):
            if not skipping:
                out.append(note)
            skipping = True
            continue
        if skipping:
            if _NEXT_HEADING.match(line) and not heading.match(line):
                skipping = False
                out.append(line)
            continue
        out.append(line)
    return "\n".join(out)


def strip_token_glosses(reply: str) -> str:
    """Drop individual token-gloss lines, leaving one honest note in their place.

    Line-wise rather than section-wise because the laundered version has no section to
    remove — the glosses sit under an innocuous heading, or under none at all.
    """
    out, noted = [], False
    for line in reply.splitlines():
        if _TOKEN_GLOSS.match(line):
            if not noted:
                out.append(_NO_WBW_NOTE)
                noted = True
            continue
        out.append(line)
    return "\n".join(out)


def strip_grammar(reply: str) -> str:
    """Remove a grammatical/morphological analysis and say plainly we don't store one.

    Deterministic for the same reason as strip_word_by_word: the model is convincingly
    right often enough that only a rule, not a judgement, can hold the line. See
    _GRAMMAR_HEADING for why this applies to every verse rather than only unglossed ones.
    """
    return _strip_section(reply, _GRAMMAR_HEADING, _NO_GRAMMAR_NOTE)


def strip_word_by_word(reply: str) -> str:
    """Remove a word-by-word section and say plainly that we don't have one.

    Deterministic rather than model-dependent, because the model is CONVINCINGLY right:
    asked for Yoga Sutras 1.3 it produced 'draṣṭuḥ — of the Seer', which is a correct
    gloss it recalled from training, not from the KB (that sutra has no stored
    word_meanings). Correct-from-memory is exactly the case the zero-hallucination
    guarantee exists to catch — nothing verified it, and the next one may be wrong.
    """
    out, skipping = [], False
    for line in reply.splitlines():
        if _WBW_HEADING.match(line):
            if not skipping:
                out.append(_NO_WBW_NOTE)
            skipping = True
            continue
        if skipping:
            if _NEXT_HEADING.match(line):
                skipping = False
                out.append(line)
            continue
        out.append(line)
    return "\n".join(out)


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
