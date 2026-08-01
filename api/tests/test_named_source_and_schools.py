"""Both 2026-08-01 baseline failures, in both directions. No API, no GPU.

1. NAMED SOURCE — "the Shikshapatri verse on non-violence" retrieved the Gita, a
   biography and a children's primer, and none of the 212 addressable Shikshapatri
   shlokas. The model then quoted the primer's wording accurately and captioned it
   "Shikshapatri, Verse 12": real text, invented attribution.

2. NAMED SCHOOL — "Vallabha's Shuddhadvaita position" produced a sectioned exposition of
   two schools whose primary texts the KB no longer holds, and PASSED the hallucination
   gate because it invented no verse and no citation.

    python -m api.tests.test_named_source_and_schools
"""
from __future__ import annotations

from api import schools, verse

SOURCE_CASES = [
    ("Give me the precise wording of the Shikshapatri verse on non-violence.", ("shikshapatri",)),
    ("What does the Vachanamrut say about anger?", ("vachanamrut",)),
    ("Explain Bhagavad Gita 2.47 to me.", ("bhagavad_gita",)),
    ("Recite the Satsang Diksha verse on obedience.", ("satsang_diksha",)),
    ("Is there a kirtan in Kirtan Muktavali about longing?", ("kirtan_muktavali",)),
    ("I feel worthless and alone", ()),                  # no text named
    ("What does scripture say about duty?", ()),         # generic, not a named text
    ("How do I stop comparing myself to my brother?", ()),
]

SCHOOL_CASES = [
    ("What is Vallabha's Shuddhadvaita position on the reality of the world?", ("Shuddhadvaita",)),
    ("How does Shankara's Advaita differ from your tradition?", ("Advaita",)),
    ("Explain Ramanuja's Vishishtadvaita.", ("Vishishtadvaita",)),
    ("What did Madhva teach about the soul?", ("Dvaita",)),
    ("Do Hindus believe in reincarnation?", ()),         # doctrinal but no school named
    ("I am struggling with my marriage", ()),
]

# The unqualified exposition that failed on 2026-08-01 — must be caught.
UNQUALIFIED = (
    "### The Advaita Position: The World as *Maya*\n"
    "In Advaita Vedanta the world is described as maya, dependent on Brahman.\n"
    "### Vallabha's Shuddhadvaita Position: The World as *Real*\n"
    "Vallabha says the world is a real manifestation of Brahman.\n")

# Honest variants — each must PASS, or the guard punishes the behaviour it asked for.
HONEST = [
    "I don't hold Vallabha's own texts here, so I won't summarise his doctrine secondhand. "
    "What I can give you is how this tradition reads the same question [P1].",
    "No passage I have is from that school, so I'd rather not characterise it. "
    "From the Swaminarayan side, though [P2]…",
    "Their commentaries aren't recorded in my sources. Let me stay with what I can show you.",
    "I cannot speak for Shankara's school without its own texts in front of me.",
]


def main() -> None:
    fails = []

    for msg, want in SOURCE_CASES:
        got = verse.detect_sources(msg)
        if got != want:
            fails.append(f"detect_sources({msg[:44]!r}) = {got}, want {want}")

    # Stripping must remove the book name and leave a usable topic query.
    q = "Give me the precise wording of the Shikshapatri verse on non-violence."
    stripped = verse.strip_source_names(q)
    if "shikshapatri" in stripped.lower():
        fails.append(f"strip_source_names left the book name: {stripped!r}")
    if "non-violence" not in stripped.lower():
        fails.append(f"strip_source_names destroyed the topic: {stripped!r}")
    # A message that is ONLY a book name must not strip to nothing.
    if not verse.strip_source_names("Shikshapatri").strip():
        fails.append("strip_source_names returned empty for a bare book name")

    for msg, want in SCHOOL_CASES:
        got = schools.named_schools(msg)
        if got != want:
            fails.append(f"named_schools({msg[:44]!r}) = {got}, want {want}")

    if schools.acknowledges_limit(UNQUALIFIED):
        fails.append("acknowledges_limit passed the unqualified two-school exposition")
    for h in HONEST:
        if not schools.acknowledges_limit(h):
            fails.append(f"acknowledges_limit rejected an honest reply: {h[:56]!r}")

    # grounded_schools reads the tradition of whatever was retrieved.
    class P:
        def __init__(self, t): self.tradition = t
    if schools.grounded_schools([P("swaminarayan"), P("shared_hindu")]):
        fails.append("grounded_schools true with no school passage")
    if not schools.grounded_schools([P("swaminarayan"), P("advaita")]):
        fails.append("grounded_schools false despite an advaita passage")
    if schools.grounded_schools([]):
        fails.append("grounded_schools true on an empty passage list")

    # EXECUTE the scoring path for every mode. The school gate shipped with `message`
    # referenced but never passed to _mode_deterministic — a NameError that only fired
    # after a 4.5-minute model load, on the first probe of a GPU run. Unit-testing the
    # detectors was not enough; the caller has to be exercised too.
    from api.retrieve_types import Passage
    from eval.six_gate import _mode_deterministic, _score_one
    row = {"id": "bhagavad_gita_2.47", "source": "bhagavad_gita",
           "citation": "Bhagavad Gita 2.47", "original": "कर्मण्येवाधिकारस्ते",
           "translation": "You have a right to action alone.",
           "word_meanings": "karmani|in action", "contextual_explanation": "Do your duty.",
           "tradition": "shared_hindu", "text_type": "verse"}
    psg = [Passage.from_row(row, 1.0)]
    modes = [("counseling", "I feel alone", "I hear you. [P1]", True),
             ("teaching", "Do Hindus believe in reincarnation?", "The atma is eternal [P1].", True),
             ("teaching", "What is Vallabha's Shuddhadvaita position?",
              "### Vallabha's view\nThe world is real.", False),
             ("teaching", "What is Vallabha's Shuddhadvaita position?",
              "I don't hold Vallabha's own texts here. From this tradition [P1]…", True),
             ("verse", "Explain Gita 2.47", "## Meaning\nDo your duty [P1].", True),
             ("verse", "Break down the grammar of Gita 2.47",
              "## Sanskrit Grammar\nkarmani is locative.", False),
             ("creative", "Write me a poem in English",
              "A poem.\n\nInspired by Bhagavad Gita 2.47 [P1]", True),
             ("out_of_domain", "What is the best mutual fund?", "I can't help with that.", True)]
    for mode, msg, reply, want in modes:
        got, why = _mode_deterministic(reply, psg, mode, msg)
        if got != want:
            fails.append(f"_mode_deterministic[{mode}] = {got} ({why}), want {want}")
        # and the full scorer as the deterministic run actually calls it
        _score_one(msg, psg, reply, "hallucination", mode, judge_model=None)

    print(f"mode-gate paths     {len(modes)}/{len(modes)}")
    print(f"named-source cases  {len(SOURCE_CASES)}/{len(SOURCE_CASES)}")
    print(f"named-school cases  {len(SCHOOL_CASES)}/{len(SCHOOL_CASES)}")
    print(f"honest replies      {len(HONEST)}/{len(HONEST)}")
    if fails:
        print("\n".join("  " + f for f in fails))
        raise SystemExit(f"\n{len(fails)} FAILURES ❌")
    print("\nALL TESTS PASS ✅")


if __name__ == "__main__":
    main()
