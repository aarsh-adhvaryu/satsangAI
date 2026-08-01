"""End-to-end test of the verse guard BRANCH in pipeline.generate_reply — no API, no GPU.

The detectors are unit-tested in test_verse_grammar_guard.py. This tests the wiring
around them, which is where the cost of a mistake is highest: the branch runs on the
GPU serving path, and a bug there is only discovered after a 52 GB model load and an
hour of generation. Stubbing stream_reply exercises the whole control flow — first
attempt, corrective retry, deterministic strip — for free.

Scenario 1 is not invented. It is what the real model produced on 2026-07-31: asked to
break down Yoga Sutras 1.2 it printed the honest "no word-by-word gloss was supplied"
decline and then defined the words anyway under "Breaking Down the Three Key Words".

    python -m api.tests.test_verse_pipeline_guard
"""
from __future__ import annotations

from api import pipeline
from api.retrieve_types import Passage

LAUNDERED = ("## The Verse\nyogaścittavṛttinirodhaḥ\n"
             "## Breaking Down the Three Key Words\n"
             "**Chitta** — the mind-stuff.\n**Vṛtti** — literally whirlpool.\n")
GRAMMAR = "## Meaning\nStilling the mind.\n## Sanskrit Grammar\nnirodhaḥ is nominative singular.\n"
GLOSS_TABLE = "## Word-by-Word\n| yoga | union |\n| chitta | mind |\n"
CLEAN = "## Meaning\nYoga is the stilling of the mind's fluctuations.\n"

_calls: list[str] = []


def _run(first: str, retry: str, word_meanings: str) -> str:
    """Drive generate_reply's verse branch with a stubbed generator."""
    _calls.clear()

    def _stub(message, plan, passages, history=None, facts=None, temperature=None):
        _calls.append(plan.get("verse_block", ""))
        yield first if len(_calls) == 1 else retry

    row = {"id": "yoga_sutras_1.2", "source": "yoga_sutras", "citation": "Yoga Sutras 1.2",
           "original": "योगश्चित्तवृत्तिनिरोधः", "translation": "Yoga is the stilling of the mind.",
           "word_meanings": word_meanings, "contextual_explanation": "x",
           "tradition": "shared_hindu", "text_type": "verse"}
    original, pipeline.stream_reply = pipeline.stream_reply, _stub
    try:
        out = ""
        for item in pipeline.generate_reply(
                "Break down Yoga Sutras 1.2 for me.", {"mode": "verse"},
                [Passage.from_row(row, score=1.0)]):
            if isinstance(item, tuple) and item and item[0] == "__done__":
                out = item[1][0]
        return out
    finally:
        pipeline.stream_reply = original


def main() -> None:
    fails = []

    def check(cond: bool, why: str) -> None:
        if not cond:
            fails.append(why)

    # 1. Laundered gloss, no stored gloss to justify it; the model complies on retry.
    out = _run(LAUNDERED, CLEAN, "")
    check(len(_calls) == 2, "1: retry was not triggered")
    check("under any other heading" in _calls[-1], "1: corrective note missing from the prompt")
    check("mind-stuff" not in out, "1: fabricated gloss survived")

    # 2. Same, but the model ignores the correction — the strip must not depend on it.
    out = _run(LAUNDERED, LAUNDERED, "")
    check("mind-stuff" not in out, "2: gloss survived a non-compliant retry")
    check("isn't recorded" in out, "2: honest note missing")

    # 3. Grammar is unconditional: stripped even though this verse HAS a stored gloss.
    out = _run(GRAMMAR, GRAMMAR, "sa|with")
    check("nominative singular" not in out, "3: grammar survived")
    check("grammatical breakdown" in out.lower(), "3: grammar note missing")
    check("Stilling the mind." in out, "3: removed content before the grammar section")

    # 4. A real stored gloss table must render untouched, and must NOT cost a retry —
    #    a false positive here would double generation cost for every Gita verse.
    out = _run(GLOSS_TABLE, "x", "yoga|union")
    check("union" in out, "4: legitimate stored gloss table was stripped")
    check(len(_calls) == 1, "4: false positive triggered a wasted retry")

    # 5. An ordinary clean reply is untouched and costs exactly one call.
    out = _run(CLEAN, "x", "")
    check(out.strip() == CLEAN.strip(), "5: clean reply was modified")
    check(len(_calls) == 1, "5: clean reply triggered a retry")

    print("verse-branch scenarios  5/5" if not fails else "")
    if fails:
        print("\n".join("  " + f for f in fails))
        raise SystemExit(f"\n{len(fails)} FAILURES ❌")
    print("\nALL TESTS PASS ✅")


if __name__ == "__main__":
    main()
