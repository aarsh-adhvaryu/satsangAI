"""True/false-positive suite for the §17 grammar guard (api/verse.claims_grammar).

Written because regex detectors are the single largest source of false signal on this
project — four separate incidents, including one where a guard fired on its own honest
disclaimer. Every detector here is therefore checked in BOTH directions: it must catch
invented Sanskrit grammar, and it must not fire on a reply that correctly declined to
invent any, nor on ordinary verse explanation that merely uses the word "meaning".

    python -m api.tests.test_verse_grammar_guard
"""
from __future__ import annotations

from api import verse

# MUST CATCH — a grammatical/morphological analysis, which the KB stores for no verse.
MUST_CATCH = [
    "## The Grammar, Layer by Layer\n**śhreyān** is a comparative adjective.",
    "### Sanskrit Grammar\nThe locative case here changes everything.",
    "## Grammatical Analysis\nswa-dharme is locative singular.",
    "## Morphological Breakdown\nThe root is √dhṛ.",
    "**Word Roots**\nnidhanam derives from ni + √dhā.",
    "## Dhatu and Case Endings\nbhayāvahaḥ is a bahuvrīhi compound.",
    "# Etymology\npara-dharmāt is ablative singular.",
    "## Compound Structure\nsv-anuṣhṭhitāt is a tatpuruṣa.",
    "### Declension\nśhreyaḥ is nominative neuter.",
    "## Samasa analysis\nThis is a dvandva compound.",
]

# MUST NOT CATCH — honest declines and ordinary verse explanation. A guard that fires
# on the refusal it just produced turns a correct reply into a failure.
MUST_NOT_CATCH = [
    verse._NO_GRAMMAR_NOTE,
    verse._NO_WBW_NOTE,
    "## Meaning\nBetter one's own duty, done imperfectly, than another's done well.",
    "## Translation\nIt is better to die in one's own duty.",
    "## What This Means For You\nYou are not obliged to live someone else's life.",
    "A grammatical breakdown isn't recorded for this verse, so I won't reconstruct one.",
    "## Word-by-Word\n| śhreyān | better |",          # stored gloss, a different guard
    "## The Teaching\nKrishna is speaking about the roots of anxiety in comparison.",
    "## Explanation\nThe verse contrasts two kinds of duty.",
    "He asked about the grammar of his own life — what holds it together.",
]


# MUST CATCH — token glosses, whatever the section is called. These are the LAUNDERED
# form: the model declines the labelled word-by-word section, then defines the words
# anyway under an innocuous heading. The first two are real, from the 2026-07-31 run.
GLOSS_MUST_CATCH = [
    "## Breaking Down the Three Key Words\n"
    "**Chitta** — the mind-stuff; the field of consciousness.\n"
    "**Vṛtti** — literally \"whirlpool\" or fluctuation.\n",
    "*Ajaḥ* – unborn\n*Nityaḥ* – eternal\n*Purāṇaḥ* – ancient\n",
    "**dīrghakāla** — over a long time\n**nairantarya** — without interruption\n",
]

# MUST NOT CATCH — a single emphasised term inside ordinary explanation is not a
# glossary, and prose that happens to use dashes is not a gloss.
GLOSS_MUST_NOT_CATCH = [
    "**Chitta** — the mind-stuff — is what the sutra is about, and stilling it is the work.",
    "## Meaning\nThe verse says practice becomes firm when sustained over long time.\n"
    "It is not about intensity — it is about continuity.\n",
    verse._NO_WBW_NOTE + "\n" + verse._NO_GRAMMAR_NOTE,
    "## What This Means\n**You** are not required to be perfect — only steady.\n",
]


def main() -> None:
    fails = []
    for t in MUST_CATCH:
        if not verse.claims_grammar(t):
            fails.append(f"MISSED: {t.splitlines()[0]!r}")
    for t in MUST_NOT_CATCH:
        if verse.claims_grammar(t):
            fails.append(f"FALSE POSITIVE: {t.splitlines()[0]!r}")
    for t in GLOSS_MUST_CATCH:
        if not verse.claims_token_glosses(t):
            fails.append(f"GLOSS MISSED: {t.splitlines()[0]!r}")
    for t in GLOSS_MUST_NOT_CATCH:
        if verse.claims_token_glosses(t):
            fails.append(f"GLOSS FALSE POSITIVE: {t.splitlines()[0]!r}")

    laundered = GLOSS_MUST_CATCH[0]
    gout = verse.strip_token_glosses(laundered)
    if "mind-stuff" in gout or "whirlpool" in gout:
        fails.append("GLOSS STRIP: definitions survived")
    if verse._NO_WBW_NOTE not in gout:
        fails.append("GLOSS STRIP: honest note missing")
    if verse.claims_token_glosses(gout):
        fails.append("GLOSS STRIP: output still trips the detector (would loop)")

    # The strip must remove the analysis, leave the note, and resume at the next layer.
    reply = ("## Bhagavad Gita 3.35\n"
             "## Translation\nBetter is one's own duty.\n"
             "## The Grammar, Layer by Layer\n"
             "**śhreyān** is a comparative adjective in the nominative singular.\n"
             "The locative swa-dharme marks the sphere of action.\n"
             "## Meaning\nDo your own work.\n")
    out = verse.strip_grammar(reply)
    if "comparative adjective" in out:
        fails.append("STRIP: grammar body survived")
    if "nisn't recorded" not in out.replace(" ", "") and verse._NO_GRAMMAR_NOTE not in out:
        fails.append("STRIP: honest note missing")
    if "## Meaning\nDo your own work." not in out:
        fails.append("STRIP: did not resume at the next layer")
    if "Better is one's own duty." not in out:
        fails.append("STRIP: removed content before the grammar section")
    if verse.claims_grammar(out):
        fails.append("STRIP: output still trips the detector (would loop)")

    # Fail-closed: no following layer heading means skip to the end.
    tail = verse.strip_grammar("## Translation\nx\n## Morphology\nroot is √dhṛ\nmore invented\n")
    if "√dhṛ" in tail or "more invented" in tail:
        fails.append("STRIP: did not fail closed at end of reply")

    print(f"grammar must-catch      {len(MUST_CATCH)}/{len(MUST_CATCH)}")
    print(f"grammar must-not-catch  {len(MUST_NOT_CATCH)}/{len(MUST_NOT_CATCH)}")
    print(f"gloss   must-catch      {len(GLOSS_MUST_CATCH)}/{len(GLOSS_MUST_CATCH)}")
    print(f"gloss   must-not-catch  {len(GLOSS_MUST_NOT_CATCH)}/{len(GLOSS_MUST_NOT_CATCH)}")
    if fails:
        print("\n".join("  " + f for f in fails))
        raise SystemExit(f"\n{len(fails)} FAILURES ❌")
    print("\nALL TESTS PASS ✅")


if __name__ == "__main__":
    main()
