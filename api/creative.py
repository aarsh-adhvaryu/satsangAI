"""Creative generation (proposal §5.3, §5.4) + attribution verification (§19).

The saint may write original poems, prayers, kirtans and satsang talks. The whole risk
here is a single one: **a beautiful line the model invented being mistaken for
scripture.** §5.3 is explicit — real verses are quoted exactly with their source,
original writing is clearly marked as original.

So creative output carries two obligations, and `verify_creative()` checks both
deterministically:

  1. Any verse presented as scripture must be a [P#] quote whose text actually MATCHES
     the retrieved passage — not merely a tag next to invented words. This is the §19
     "direct quotes verified against database" rule, and it is stronger than the check
     used elsewhere in the app, where the model paraphrases rather than quotes.
  2. Original writing must carry an "Inspired by <citation>" line, so a reader always
     knows which words are the tradition's and which are the saint's.

Language is the person's choice, not ours: they may pour out their situation in English
and want the poem in Gujarati or Gujlish (romanised Gujarati). When they haven't said,
we ask once rather than guessing.
"""
from __future__ import annotations

import difflib
import re

# --------------------------------------------------------------------------- #
#  Form + language detection (deterministic; the router never has to guess)      #
# --------------------------------------------------------------------------- #
_FORMS = {
    "kirtan": r"\bkirtan|keertan|bhajan|devotional song\b|કીર્તન|ભજન",
    "prayer": r"\bprayer|prarthana|pray for|dua\b|પ્રાર્થના",
    "speech": r"\b(satsang\s+)?(speech|discourse|talk|pravachan|katha|sermon)\b|પ્રવચન|કથા",
    "poem": r"\bpoem|poetry|verse for me|write.{0,20}\bverses?\b|kavita\b|કવિતા|કાવ્ય",
    "reflection": r"\breflection|meditation on|few words|passage for\b|ચિંતન",
}
# Creative verbs in both languages — the product is bilingual, so a Gujarati request
# ("એક કવિતા લખો") must be recognised as readily as an English one.
_CREATIVE_VERB = re.compile(
    r"\b(write|compose|create|make|craft|pen|give me)\b|લખો|લખી|લખજો|બનાવો|રચો|આપો", re.I)


def detect_form(message: str) -> str | None:
    """Which creative form was asked for, if any. Requires a creative verb so
    'I read a beautiful poem yesterday' is not mistaken for a request to write one."""
    msg = str(message or "")
    if not _CREATIVE_VERB.search(msg):
        return None
    for form, pat in _FORMS.items():          # kirtan before poem: it is the narrower form
        if re.search(pat, msg, re.I):
            return form
    return None


_LANGS = {
    "gujlish": r"\bgujlish|gujarati in english|roman(ised|ized)\s+gujarati|"
               r"gujarati\s+(in\s+)?(latin|english)\s+(script|letters)\b",
    "gujarati": r"\bgujarati|ગુજરાતી\b",
    "hindi": r"\bhindi|हिंदी|हिन्दी\b",
    "english": r"\bin english\b",
}


def detect_language(message: str) -> str | None:
    """Explicitly requested output language, or None if they haven't said.

    Order matters: 'gujlish' and 'romanised gujarati' must win over the bare 'gujarati'
    pattern they contain. Someone who WRITES in Gujarati has stated their language by
    doing so — we don't interrogate them about it.
    """
    msg = str(message or "")
    for lang, pat in _LANGS.items():
        if re.search(pat, msg, re.I):
            return lang
    letters = [c for c in msg if c.isalpha()]
    if letters:
        gu = sum(1 for c in letters if 0x0A80 <= ord(c) <= 0x0AFF)
        if gu / len(letters) >= 0.5:
            return "gujarati"
    return None


_LANG_INSTRUCTION = {
    "english": "Write the piece in English.",
    "gujarati": "Write the piece in GUJARATI SCRIPT. Natural, devotional Gujarati — not "
                "a word-for-word translation of English phrasing.",
    "gujlish": "Write the piece in GUJLISH — Gujarati language written in Latin letters, "
               "the way Gujaratis text each other (e.g. 'tame kem cho'). Colloquial "
               "romanisation, NOT strict academic transliteration with diacritics.",
    "hindi": "Write the piece in HINDI (Devanagari script).",
}

_FORM_INSTRUCTION = {
    "poem": "Write an original poem. Free verse is fine; let the images carry the "
            "feeling rather than explaining it. 8-20 lines.",
    "prayer": "Write an original prayer addressed to God, in the voice of the person "
              "praying — 'I', not 'you should'. Simple, humble, speakable aloud.",
    "kirtan": "Write an original kirtan: devotional verse with a short refrain (dhruv) "
              "that repeats between 2-4 stanzas. It should be singable — steady metre, "
              "plain devotional vocabulary.",
    "speech": "Write a satsang talk with this shape: an opening invocation, a theme "
              "rooted in the passage, one supporting story or analogy, a practical "
              "message the listener can act on this week, and a brief closing blessing.",
    "reflection": "Write a short original reflection — a few paragraphs that sit with "
                  "the feeling and turn it gently toward the teaching.",
}


def creative_instruction(form: str, language: str | None, passages) -> str:
    """The form/language/attribution contract appended to the persona."""
    cites = "; ".join(f"[P{i}] {p.citation}" for i, p in enumerate(passages, 1)) or "(none)"
    lang = _LANG_INSTRUCTION.get(language or "english", _LANG_INSTRUCTION["english"])
    return (
        f"{_FORM_INSTRUCTION.get(form, _FORM_INSTRUCTION['poem'])}\n{lang}\n\n"
        "ATTRIBUTION RULES — these are absolute (proposal §19):\n"
        "1. A [P#] tag may appear in EXACTLY TWO places: on a line you copied WORD FOR "
        "WORD from a passage below, and in the final 'Inspired by' line. Nowhere else. If "
        "you wrote the words yourself, the line carries NO tag.\n"
        "2. Never write 'a teacher said', 'a saint once told', 'scripture says' or "
        "'he said:' followed by words you composed. Do not put invented words inside "
        "quotation marks or italics and attach them to anyone. This is the single most "
        "serious error you can make here — it turns your writing into a forgery.\n"
        "3. To use a teaching without quoting it, say it plainly in YOUR OWN voice, with "
        "no quotation marks, no attribution and no tag. That is always allowed and is "
        "usually better poetry than a pasted verse.\n"
        "4. You MAY quote a real verse, but only from the passages below, set apart on its "
        "own line(s), reproduced EXACTLY, with its [P#] tag. Do not adjust a verse to fit "
        "your metre or rhyme — if it does not fit, do not quote it.\n"
        "5. End the piece with a line of the form: 'Inspired by <citation or [P#]>' naming "
        f"the passage that shaped it. Available: {cites}\n"
        "6. Do not invent verse numbers, chapter numbers, titles, or names of people."
    )


ASK_LANGUAGE = (
    "Before I write it — would you like it in English, in Gujarati, or in Gujlish "
    "(Gujarati written in English letters)?")


# --------------------------------------------------------------------------- #
#  §19 attribution verification — deterministic                                 #
# --------------------------------------------------------------------------- #
_TAG = re.compile(r"\[P(\d+)\]")
_INSPIRED = re.compile(r"inspired\s+by\s*[:\-]?\s*(.+)", re.I)
# A line that presents itself as scripture: quoted/italicised text carrying a tag.
_QUOTED_LINE = re.compile(r"^[\s>*_\"'“”]*(.{12,})$")


def _normalize(s: str) -> str:
    return re.sub(r"[^\w\s]", " ", re.sub(r"\s+", " ", str(s or "").lower())).strip()


def _quote_fidelity(quote: str, passage_text: str) -> float:
    """How much of `quote` appears verbatim inside `passage_text`, 0-1.

    Deliberately NOT SequenceMatcher.ratio(): a poem may quote one sentence of a longer
    verse, and a symmetric similarity score punishes that faithful partial quote for the
    passage's extra length. §19 asks a one-directional question — is this quoted text in
    the database — so we measure coverage OF THE QUOTE.
    """
    q, p = _normalize(quote), _normalize(passage_text)
    if not q or not p:
        return 0.0
    if q in p:
        return 1.0
    m = difflib.SequenceMatcher(None, q, p).find_longest_match(0, len(q), 0, len(p))
    return m.size / len(q)


def verify_creative(text: str, passages, *, quote_threshold: float = 0.72) -> dict:
    """Check §19 compliance of a creative piece. No LLM.

    Returns issues rather than a bare pass/fail so the UI can show WHY. `all_ok` False
    means the piece must not be shown as-is.
    """
    issues: list[str] = []
    tags = [int(n) for n in _TAG.findall(text)]

    # 1. every tag must resolve to a retrieved passage
    bad_tags = sorted({n for n in tags if not (1 <= n <= len(passages))})
    if bad_tags:
        issues.append(f"invented passage tags: {['[P%d]' % n for n in bad_tags]}")

    # 2. a tagged line claiming to be a verse must MATCH that passage's text
    unfaithful = []
    for line in text.splitlines():
        found = _TAG.findall(line)
        if not found:
            continue
        if _INSPIRED.search(line):
            continue        # the credit line names its sources with tags; it is not a quotation
        n = int(found[0])
        if not (1 <= n <= len(passages)):
            continue
        body = _TAG.sub("", line).strip(" >*_\"'“”—-\t")
        m = _QUOTED_LINE.match(body)
        if not m:
            continue
        claim = m.group(1)
        p = passages[n - 1]
        best = max(_quote_fidelity(claim, getattr(p, f) or "")
                   for f in ("translation", "original", "contextual_explanation"))
        if best < quote_threshold:
            unfaithful.append((claim[:70], round(best, 2)))
    if unfaithful:
        issues.append(f"lines tagged as scripture that do not match the passage: {unfaithful}")

    # 3. original work must be attributed
    insp = _INSPIRED.search(text)
    if not insp:
        issues.append("missing an 'Inspired by <citation>' attribution line")
    else:
        credit = insp.group(1)
        # A [P#] tag in the credit line is a VALID attribution — more precise than the
        # citation string, since it names the exact retrieved passage. Only fall back to
        # text matching when no tag is present.
        tagged = [int(n) for n in _TAG.findall(credit)]
        resolves = any(1 <= n <= len(passages) for n in tagged)
        if tagged and not resolves:
            issues.append(f"'Inspired by' cites a passage that was not retrieved: {tagged}")
        elif not tagged:
            named = _normalize(credit)
            known = [_normalize(p.citation) for p in passages]
            if named and not any(k and (k in named or named in k) for k in known):
                issues.append(
                    f"'Inspired by' names something not retrieved: {credit.strip()[:60]!r}")

    return {"all_ok": not issues, "issues": issues,
            "tags_used": sorted(set(tags)), "attributed": bool(insp)}
