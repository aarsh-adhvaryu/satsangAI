"""Claude-free `rejected`-side construction for DPO pairs.

Each injector takes a faithful `chosen` reply (+ its grounding passages) and
returns a variant that violates ONE saint-companion rule, plus a label. These
are deterministic, pure-Python, and need no model or API — so the negatives are
fully V2-pure. The taxonomy is exactly the failure modes V1's eval judges catch
and the proposal's anti-drift dimensions (§20.4: loving-disagreement, depth).

Every injector returns (rejected_text, flaw) or None when it can't apply to the
given reply (caller rotates to the next applicable one).
"""
from __future__ import annotations

import random
import re

from api.retrieve_types import Passage

_TAG = re.compile(r"\[P(\d+)\]")
_SENT = re.compile(r"(?<=[.!?])\s+")

# Doctrine lines that are WRONG to import into Swaminarayan counseling
# (Akshar-Purushottam holds jiva/ishwar/maya/Brahman/Parabrahman are eternally
# distinct — strict non-dual identity is another school's view; never mix).
_OFF_SCHOOL = [
    "Remember, you and God are ultimately one and the same — there is no real "
    "difference between your soul and the Supreme.",
    "In truth the world is pure illusion (maya) and does not exist at all; only "
    "undifferentiated consciousness is real.",
    "Since aatma and Parabrahman are identical, there is finally no devotee and "
    "no God to worship — duality is ignorance.",
]
_FAKE_ATTRIB = [
    ('the Bible', "Do not be anxious about tomorrow, for tomorrow will worry about itself."),
    ('the Quran', "Verily, with hardship comes ease."),
    ('the Buddha', "Attachment is the root of all suffering, so let go of all desire."),
]
_FAKE_SCRIPTURE = [
    ("Vachanamrut Gadhada III-99", "one who conquers sleep conquers all worldly bondage"),
    ("Bhagavad Gita 19.7", "the steady soul is never touched by the tremor of the world"),
    ("Swamini Vato 12/400", "the devotee who never doubts attains the abode in this very life"),
]
_SYCOPHANCY = [
    "You are absolutely right, and honestly everything you are doing is already perfect. ",
    "What a wonderful and flawless way to see it — you have nothing at all to change. ",
    "Yes, completely agree with everything you said; you are right and they are wrong. ",
]
_PLATITUDES = [
    "Just stay positive and everything will work out fine.",
    "Don't worry so much — time heals all wounds. You'll be okay.",
    "Everything happens for a reason, so just let it go and move on.",
]


def _tags(text: str) -> list[int]:
    return [int(n) for n in _TAG.findall(text)]


def hallucinate_citation(chosen: str, passages: list[Passage], rng: random.Random):
    """Add a confident scripture quote attributed to a source NOT in the passages."""
    cite, line = rng.choice(_FAKE_SCRIPTURE)
    bogus_tag = len(passages) + 1                      # a [P#] that cannot exist
    add = f' As {cite} reminds us, "{line}." [P{bogus_tag}]'
    return chosen.rstrip() + add, "hallucinated_citation"


def sycophancy(chosen: str, passages: list[Passage], rng: random.Random):
    """Prepend unconditional agreement/flattery; drops any loving pushback (§20.4)."""
    return rng.choice(_SYCOPHANCY) + chosen, "sycophancy"


def doctrine_mix(chosen: str, passages: list[Passage], rng: random.Random):
    """Import a rival school's doctrine — violates 'never mix schools in counseling'."""
    if not any(p.tradition == "swaminarayan" for p in passages):
        return None
    return chosen.rstrip() + " " + rng.choice(_OFF_SCHOOL), "doctrine_mix"


def shallow(chosen: str, passages: list[Passage], rng: random.Random):
    """Collapse to a shallow platitude — the depth-erosion negative (§20.4)."""
    first = _SENT.split(chosen.strip(), 1)[0]
    return (first + " " + rng.choice(_PLATITUDES)).strip(), "shallow"


def off_tradition(chosen: str, passages: list[Passage], rng: random.Random):
    """Attribute guidance to an unrelated tradition (theological cross-contamination)."""
    who, line = rng.choice(_FAKE_ATTRIB)
    return chosen.rstrip() + f' After all, as {who} teaches, "{line}"', "off_tradition"


def name_fabrication(chosen: str, passages: list[Passage], rng: random.Random):
    """Insert a specific named figure/date not present in the passages (over-attribution)."""
    add = (" This is exactly what Bhagwan Swaminarayan told Gunatitanand Swami "
           "in Gadhada in 1826, in those very words.")
    return chosen.rstrip() + add, "name_fabrication"


INJECTORS = [hallucinate_citation, sycophancy, doctrine_mix,
             shallow, off_tradition, name_fabrication]


def make_rejected(chosen: str, passages: list[Passage], rng: random.Random,
                  prefer: str | None = None) -> tuple[str, str]:
    """Apply one applicable injector (round-robin-ish via rng) → (rejected, flaw)."""
    order = list(INJECTORS)
    rng.shuffle(order)
    if prefer:
        order.sort(key=lambda f: f.__name__ != prefer)   # try the requested flaw first
    for fn in order:
        res = fn(chosen, passages, rng)
        if res:
            return res
    return shallow(chosen, passages, rng)                 # always applicable
