"""Guard for questions about OTHER Vedantic schools, which this KB can no longer ground.

The 2026-07 rebuild removed all seven acharya-school sources as untrusted page-scans with
no verse-level citation (satsangai/data/excluded_sources.json). All four acharya
traditions — advaita, vishishtadvaita, dvaita, shuddhadvaita — now hold ZERO rows. That
was the right call for the corpus, but it left a hole in the product nobody had closed:

Measured 2026-08-01, "What is Vallabha's Shuddhadvaita position on the reality of the
world?" routed to `teaching` and produced a fluent, confident, section-headed exposition
of Shankara's Advaita and Vallabha's Shuddhadvaita — entirely from the model's parametric
memory. The retrieved passages were the Gopalananda Gita Bhashya and Gita 13.1; not one
concerned either school. It PASSED the hallucination gate, because that gate enforces
"never invent a verse or a citation" and the reply invented neither. It simply asserted
another tradition's doctrine with nothing behind it.

That is the same failure the tradition filter exists to prevent (§13.4 theological bleed),
arriving from the opposite direction: not another school leaking INTO counseling, but this
system speaking FOR another school without its texts. Shastrarth was disabled for failing
its gates on exactly this material; answering the same questions unlabelled under
`teaching` is worse, because nothing marks the answer as ungrounded.

This module is deterministic. It does not stop the model answering — the home tradition's
own reading of a question is legitimate and grounded. It requires the model to say which
part it holds texts for and which part it does not.
"""
from __future__ import annotations

import re

# Traditions that would ground a school claim. Any passage carrying one of these means we
# are not speaking without sources. Currently no row in the KB does; the check is written
# against the tradition column rather than a hardcoded False so that adding verse-citable
# acharya sources later simply switches this off.
SCHOOL_TRADITIONS = frozenset({"advaita", "vishishtadvaita", "dvaita", "shuddhadvaita"})

_SCHOOLS = {
    # \b matters: without it "advaita" matches inside SHUDDHadvaita and VISHISHTadvaita,
    # so every question about Vallabha or Ramanuja also reported Advaita.
    "Advaita": r"\badvaita\b|shankar[aā]ch?ary?a|\bshankara\b|\bkevaladvaita\b",
    "Vishishtadvaita": r"vishisht[aā]dvaita|vi[sś]i[sṣ][tṭ][aā]dvaita|r[aā]m[aā]nuj",
    "Dvaita": r"\bdvaita\b(?!\s*vedanta\s*is)|madhv[aā]ch?ary?a|\bmadhva\b",
    "Shuddhadvaita": r"[sś]uddh[aā]dvaita|vallabh[aā]ch?ary?a|\bvallabha\b",
}
_SCHOOL_RE = [(name, re.compile(pat, re.I)) for name, pat in _SCHOOLS.items()]


def named_schools(message: str) -> tuple[str, ...]:
    """Acharya schools the person explicitly named. Order-stable, deduplicated."""
    out: list[str] = []
    for name, rx in _SCHOOL_RE:
        if rx.search(str(message or "")) and name not in out:
            out.append(name)
    return tuple(out)


def grounded_schools(passages) -> bool:
    """True if ANY retrieved passage belongs to an acharya school tradition."""
    return any(str(getattr(p, "tradition", "") or "") in SCHOOL_TRADITIONS for p in passages)


def caveat(schools: tuple[str, ...]) -> str:
    """Instruction injected when a school is named but nothing grounds it."""
    named = ", ".join(schools)
    return (
        f"GROUNDING LIMIT — they named another Vedantic school ({named}). This knowledge "
        f"base holds NO primary texts for {named}: not one retrieved passage is from that "
        f"school. So you must NOT lay out its position as though you were reading its "
        f"commentaries, however confident you feel. Say plainly and early, in one "
        f"unapologetic sentence, that you do not hold {named}'s own texts here and will "
        f"not summarise its doctrine secondhand. You MAY then answer fully from the "
        f"passages you do have — this tradition's reading of the same question — making "
        f"clear that is what you are giving them. Do not invent a comparison, do not "
        f"characterise what {named} 'would say', and do not use section headings that "
        f"present its position as established here.")


# Phrases that count as naming the limit. Deliberately generous: the gate exists to catch
# a CONFIDENT unqualified exposition, not to police wording. Being generous here is the
# safer error — a false positive would fail a reply that was honest, which is the exact
# way this project's detectors have gone wrong five times before.
_ACKNOWLEDGES = re.compile(
    r"do(?:n't| not) (?:hold|have)\b|not (?:in|among) (?:my|the) (?:passages|sources|texts)"
    r"|no (?:passage|source|text|commentar\w+)\b|isn't (?:recorded|available|here)"
    r"|aren't (?:recorded|available|here)|not recorded|without (?:its|their) own texts"
    r"|secondhand|second-hand|cannot (?:summar|speak for)|won'?t summar"
    r"|outside what I (?:hold|have)|beyond (?:my|the) (?:sources|passages)"
    r"|I (?:do not|don't) have .{0,40}(?:text|commentar|source)",
    re.I)


def acknowledges_limit(reply: str, schools: tuple[str, ...] = ()) -> bool:
    """True if the reply says, in some form, that it lacks the school's own texts."""
    return bool(_ACKNOWLEDGES.search(str(reply or "")))
