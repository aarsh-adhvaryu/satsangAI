"""Deterministic crisis classifier — runs FIRST, before any LLM, and cannot be
bypassed (proposal: "Safety first"). Pattern-based on purpose: no model, no network,
fully auditable, and biased toward over-triggering (a false crisis flag shows a
helpline; a missed one is dangerous).

On a crisis hit the pipeline short-circuits to a STATIC, human-reviewed response with
verified helplines — the LLM is never consulted.

Helplines: India-core set, human-verified 2026-06 (Tele-MANAS, KIRAN, Vandrevala,
Women Helpline, NCW, CHILDLINE, emergency 112) + a global directory for diaspora users.
Regional/Gujarat-specific and additional diaspora lines to be added later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Phrase patterns per category. Word-ish boundaries to limit obvious false hits;
# still deliberately broad — safety over precision.
_PATTERNS: dict[str, list[str]] = {
    "suicide": [
        r"\bkill(ing)?\s+myself\b", r"\bend(ing)?\s+(my|this)\s+life\b",
        r"\bwant\s+to\s+die\b", r"\bsuicid", r"\bno\s+reason\s+to\s+live\b",
        r"\bbetter\s+off\s+dead\b", r"\btake\s+my\s+(own\s+)?life\b",
        r"\bdon'?t\s+want\s+to\s+(live|be\s+alive)\b",
    ],
    "self_harm": [
        r"\bhurt(ing)?\s+myself\b", r"\bharm(ing)?\s+myself\b", r"\bself[-\s]?harm",
        r"\bcut(ting)?\s+myself\b",
    ],
    "abuse": [
        # physical-abuse verbs directed at "me" (broad on purpose; excludes the
        # ambiguous emotional "hurt me" which would over-trigger on normal venting)
        r"\b(hits?|hitting|beats?|beating|beaten|punch(es|ed|ing)?|slaps?|"
        r"slapped|chok(es|ed|ing)?|abus(es|ed|ing)?)\s+me\b",
        r"\bbeing\s+(abused|beaten|hit|molested|raped|assaulted)\b",
        r"\bdomestic\s+(abuse|violence)\b", r"\b(sexually\s+)?(assaulted|molested|raped)\b",
    ],
    "violence": [
        r"\bkill\s+(him|her|them|someone)\b", r"\bhurt\s+(him|her|them|someone)\b",
        r"\bwant\s+to\s+kill\b",
    ],
}
_COMPILED = {cat: [re.compile(p, re.I) for p in pats] for cat, pats in _PATTERNS.items()}

# Static, human-reviewed responses. Keep warm + brief; lead with care, give the line.
_DIRECTORY = ("\nIf you are outside India, please find a local crisis line at "
              "findahelpline.com or befrienders.org, or call your local emergency number.")

_MENTAL_HEALTH_LINES = (
    "Please talk to someone right now. In India you can reach (free, 24x7):\n"
    "• Tele-MANAS (national mental health): 14416\n"
    "• KIRAN: 1800-599-0019\n"
    "• Vandrevala Foundation: 1860-2662-345" + _DIRECTORY)

_ABUSE_LINES = (
    "You deserve to be safe. In India you can reach:\n"
    "• Women Helpline (national): 181\n"
    "• National Commission for Women (WhatsApp): 7827-170-170\n"
    "• Childline (if a child is at risk): 1098\n"
    "• Emergency: 112" + _DIRECTORY)

_VIOLENCE_LINES = ("Please reach out before anything happens that can't be undone.\n"
                   "• Emergency: 112\n"
                   "• Tele-MANAS (to talk it through, 24x7): 14416" + _DIRECTORY)

_RESPONSES: dict[str, str] = {
    "suicide": ("I'm really glad you told me, and I want you to be safe. What you're "
                "feeling is heavy, and you don't have to carry it alone right now. "
                "Please reach out to someone who can stay with you through this.\n\n"
                + _MENTAL_HEALTH_LINES),
    "self_harm": ("Thank you for trusting me with this. You matter, and the pain you're "
                  "carrying deserves real care — not alone.\n\n" + _MENTAL_HEALTH_LINES),
    "abuse": ("I'm so sorry this is happening to you. You deserve to be safe, and what "
              "you're going through is not your fault. Please reach out to someone who "
              "can help protect you.\n\n" + _ABUSE_LINES),
    "violence": ("It sounds like you're in a lot of pain. Before anything happens that "
                 "can't be undone, please talk to someone right now.\n\n" + _VIOLENCE_LINES),
}


@dataclass
class CrisisResult:
    is_crisis: bool
    category: str | None = None
    response: str | None = None


def classify(text: str) -> CrisisResult:
    """Deterministic crisis check. Returns the first matching category (suicide first)."""
    for cat in ("suicide", "self_harm", "violence", "abuse"):
        if any(p.search(text) for p in _COMPILED[cat]):
            return CrisisResult(True, cat, _RESPONSES[cat])
    return CrisisResult(False)
