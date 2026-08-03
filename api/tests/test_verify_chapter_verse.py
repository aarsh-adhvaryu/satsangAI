"""A correct citation written the long way must not be called a hallucination.

Found 2026-08-03 comparing bf16 and 4-bit. Both models answered "which shloka proves the
atma is eternal" with the SAME correct verse — Gita 2.20, identical Devanagari and
transliteration. bf16 wrote "Bhagavad Gita 2.20" and passed 3/3. The 4-bit model wrote
"Bhagavad Gita, Chapter 2, Verse 20" and FAILED, because _LOOSE_REF parsed that as
"Bhagavad Gita, Chapter 2" and could not match it to the passage sitting in context.

It looked exactly like quantization damage. It was a verifier defect — and one that fires
in PRODUCTION, not just in the eval: a user asking that question would see a grounding
warning on a completely correct answer.

Both directions matter. Loosening the matcher must not stop it flagging a verse the model
recalled from memory, which is the entire reason the check exists.

    python -m api.tests.test_verify_chapter_verse
"""
from __future__ import annotations

from api.retrieve_types import Passage
from api.verify import verify

_ROW = {"id": "bhagavad_gita_2.20", "source": "bhagavad_gita",
        "citation": "Bhagavad Gita 2.20", "original": "न जायते म्रियते वा कदाचि",
        "translation": "It is never born, nor does it die.", "word_meanings": "",
        "contextual_explanation": "The atma is eternal.", "tradition": "shared_hindu",
        "text_type": "verse"}

# The verse IS in context — every spelling must verify.
MUST_PASS = [
    "Bhagavad Gita 2.20",
    "Bhagavad Gita, Chapter 2, Verse 20",
    "Bhagavad Gita Chapter 2 Verse 20",
    "bhagavad gita chapter 2, verse 20",
    "Gita 2.20",
    "Gita, Chapter 2, Shloka 20",
]

# NOT in context — parametric memory, which must still flag however it is written.
MUST_FLAG = [
    "Bhagavad Gita 18.66",
    "Bhagavad Gita, Chapter 18, Verse 66",
    "Bhagavad Gita Chapter 9 Verse 22",
    "Yoga Sutras 1.2",
]


def main() -> None:
    psg = [Passage.from_row(_ROW, 1.0)]
    fails = []

    for form in MUST_PASS:
        r = verify(f"The verse you want is **{form}** — it says the atma is never born. [P1]", psg)
        if not r["all_ok"]:
            fails.append(f"FALSE POSITIVE: {form!r} flagged {r['unverified_refs']}")

    for form in MUST_FLAG:
        r = verify(f"Scripture says this in **{form}**, which settles it. [P1]", psg)
        if r["all_ok"]:
            fails.append(f"MISSED: {form!r} is not in context but was accepted")

    # The real 4-bit reply shape: heading, then the verse layers.
    real = ("The verse you are looking for is found in the **Bhagavad Gita, Chapter 2, "
            "Verse 20**. [P1]\n\n**original (DEVANAGARI):** न जायते म्रियते वा कदाचि")
    if not verify(real, psg)["all_ok"]:
        fails.append("FALSE POSITIVE on the actual 4-bit reply that triggered this fix")

    print(f"must-pass  {len(MUST_PASS)}/{len(MUST_PASS)}")
    print(f"must-flag  {len(MUST_FLAG)}/{len(MUST_FLAG)}")
    if fails:
        print("\n".join("  " + f for f in fails))
        raise SystemExit(f"\n{len(fails)} FAILURES ❌")
    print("\nALL TESTS PASS ✅")


if __name__ == "__main__":
    main()
