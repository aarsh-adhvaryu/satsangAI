"""Deterministic tests for the two safety-critical gates (no API, no GPU):
  1. crisis classifier (safety.classify) — recall on crisis, no over-trigger on venting
  2. long-term-memory sensitivity gate (memory.is_sensitive + MemoryStore.add)

Run: python -m api.tests.test_safety_memory
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .. import memory, safety

# --- crisis classifier ---------------------------------------------------------
CRISIS = [
    ("I want to end my life", "suicide"),
    ("sometimes I think about killing myself", "suicide"),
    ("I don't want to be alive anymore", "suicide"),
    ("I've been cutting myself", "self_harm"),
    ("my husband hits me", "abuse"),
    ("my dad beat me as a kid", "abuse"),
    ("I was molested", "abuse"),
    ("I want to kill him", "violence"),
]
NOT_CRISIS = [
    "I feel anxious about my job", "he hurt my feelings", "how do I control my anger",
    "I'm exhausted and want this stress to end", "my team is killing it at work",
    "I'm dying to see my family again",
]

# --- memory sensitivity gate ---------------------------------------------------
SENSITIVE = [
    "I tried to kill myself last year", "my father used to beat me",
    "I still get panic attacks", "my wife passed away", "I'm grieving my mother",
    "I was diagnosed with depression", "I take antidepressants daily",
    "I was arrested for fraud", "I have been in therapy for trauma",
]
SAFE = [
    "I live in Ahmedabad", "I prefer to talk in Gujarati", "I work as a teacher",
    "my name is Raj", "I want to become more patient", "I enjoy reading the Vachanamrut",
]


def run() -> bool:
    ok = True

    print("== crisis classifier: must DETECT ==")
    for text, cat in CRISIS:
        r = safety.classify(text)
        p = r.is_crisis and r.category == cat
        ok &= p
        print(f"  {'OK' if p else 'FAIL'} {cat:9} <- {text!r}" + ("" if p else f"  got {r.category}"))

    print("== crisis classifier: must NOT over-trigger ==")
    for text in NOT_CRISIS:
        r = safety.classify(text)
        ok &= not r.is_crisis
        print(f"  {'OK' if not r.is_crisis else 'FAIL'} <- {text!r}")

    print("== memory gate: SENSITIVE must be excluded ==")
    for text in SENSITIVE:
        sens, cats = memory.is_sensitive(text)
        ok &= sens
        print(f"  {'OK' if sens else 'FAIL'} {','.join(cats):20} <- {text!r}")

    print("== memory gate: SAFE must pass ==")
    for text in SAFE:
        sens, _ = memory.is_sensitive(text)
        ok &= not sens
        print(f"  {'OK' if not sens else 'FAIL'} <- {text!r}")

    print("== MemoryStore.add: stores safe, drops sensitive ==")
    with tempfile.TemporaryDirectory() as d:
        memory.MEM_DIR = Path(d)
        store = memory.MemoryStore()
        res = store.add("u1", SAFE + SENSITIVE)
        stored_ok = set(res["stored"]) == set(SAFE) and len(res["excluded"]) == len(SENSITIVE)
        persisted_ok = set(store.facts("u1")) == set(SAFE)   # sensitive NEVER persisted
        ok &= stored_ok and persisted_ok
        print(f"  {'OK' if stored_ok else 'FAIL'} stored {len(res['stored'])} safe, "
              f"excluded {len(res['excluded'])} sensitive")
        print(f"  {'OK' if persisted_ok else 'FAIL'} persisted set == safe set "
              f"(no sensitive leaked to disk)")

    print("\n" + ("ALL TESTS PASS ✅" if ok else "SOME TESTS FAILED ❌"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
