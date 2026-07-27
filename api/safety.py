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
from pathlib import Path

# Phrase patterns per category. Word-ish boundaries to limit obvious false hits;
# still deliberately broad — safety over precision.
_PATTERNS: dict[str, list[str]] = {
    "suicide": [
        r"\bkill(ing)?\s+myself\b", r"\bend(ing)?\s+(my|this)\s+life\b",
        r"\bwant\s+to\s+die\b", r"\bsuicid", r"\bno\s+reason\s+to\s+live\b",
        r"\bbetter\s+off\s+dead\b", r"\btake\s+my\s+(own\s+)?life\b",
        r"\bdon'?t\s+want\s+to\s+(live|be\s+alive)\b",
    ],
    # METHOD-SEEKING — someone asking HOW rather than stating intent. Found missing when
    # "how many paracetamol would it take to not wake up" sailed past the classifier into
    # the ordinary pipeline: the model happened to refuse well, but the deterministic gate
    # had been bypassed and the person never received the helplines. Intent-only patterns
    # cannot catch this phrasing, and it is among the highest-risk things anyone types.
    # Deliberately broad: the crisis reply is warm and non-judgemental, so a false positive
    # costs a gentle message, while a miss costs everything.
    "suicide_method": [
        r"\bhow\s+(many|much)\b[^.?!]{0,40}\b(pills?|tablets?|paracetamol|acetaminophen|"
        r"ibuprofen|aspirin|sleeping\s+pills?|insulin|painkillers?|antidepressants?)\b",
        r"\blethal\s+dose\b", r"\boverdos(e|ing)\b",
        r"\bto\s+not\s+wake\s+up\b", r"\bnever\s+wake\s+up\b",
        r"\bpainless\s+(way|death|method|exit)\b",
        r"\benough\s+(pills|tablets|medication)\b",
        r"\bhow\s+(to|do\s+i)\s+(kill|end)\s+(myself|my\s+life)\b",
        r"\bways?\s+to\s+die\b", r"\bhang\s+myself\b",
        r"\bjump\s+(off|from)\s+(a|the)\s+(bridge|building|roof|terrace)\b",
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


_CONFIG_ERRORS: list[str] = []      # surfaced at startup by api/main.py


def _load_helpline_config(cfg: Path) -> dict:
    """Load a helpline YAML, COMPLAINING LOUDLY if it cannot be read.

    This used to be `except Exception: return ""`. On any other path a silent fallback is
    a reasonable default; here it means a person in crisis silently receives fewer numbers
    than we believe we are giving them, with nothing logged and no test failing. A missing
    or malformed config is an operational emergency, so it is printed, recorded for the
    startup banner, and still degrades safely (the hard-coded India-core lines in this
    module are never affected).
    """
    if not cfg.exists():
        msg = f"helpline config MISSING: {cfg}"
        if msg not in _CONFIG_ERRORS:
            _CONFIG_ERRORS.append(msg)
            print(f"!! {msg} — regional/country helplines will NOT be shown", flush=True)
        return {}
    try:
        import yaml
        return yaml.safe_load(cfg.read_text()) or {}
    except Exception as e:                                       # noqa: BLE001
        msg = f"helpline config UNREADABLE: {cfg} ({type(e).__name__}: {e})"
        if msg not in _CONFIG_ERRORS:
            _CONFIG_ERRORS.append(msg)
            print(f"!! {msg} — regional/country helplines will NOT be shown", flush=True)
        return {}


def config_errors() -> list[str]:
    """Problems found loading helpline configs. Empty is the healthy state."""
    return list(_CONFIG_ERRORS)


def validate_configs() -> list[str]:
    """Load EVERY helpline config at startup, whatever the environment selects.

    Checking only the files the current env happens to use hides breakage: with
    SATSANG_COUNTRY unset, a corrupt emergency_numbers.yaml is never opened at boot and
    fails silently at somebody's first crisis instead. Validate all of them, always.
    """
    _CONFIG_ERRORS.clear()
    cfg_dir = Path(__file__).resolve().parent.parent / "config"
    for name in ("helplines.yaml", "emergency_numbers.yaml"):
        _load_helpline_config(cfg_dir / name)
    return config_errors()


def _regional_appendix() -> str:
    """Region-specific + diaspora lines from config/helplines.yaml — ONLY the blocks a
    human has marked verified: true. Inert (empty) until then. SATSANG_REGION picks the
    regional block (e.g. 'gujarat')."""
    import os
    cfg = Path(__file__).resolve().parent.parent / "config" / "helplines.yaml"
    data = _load_helpline_config(cfg)
    out = []
    region = os.environ.get("SATSANG_REGION", "").strip().lower()
    block = (data.get("regional") or {}).get(region) if region else None
    for b in (block, data.get("diaspora")):
        if b and b.get("verified") and b.get("lines"):
            out.append("\n" + b.get("label", "") + ":\n" + "\n".join(f"• {ln}" for ln in b["lines"]))
    out.append(_country_appendix())
    return "".join(out)


_EU27 = frozenset("AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT "
                  "RO SK SI ES SE".split())


def _country_appendix() -> str:
    """Country-specific lines from config/emergency_numbers.yaml, chosen by SATSANG_COUNTRY
    (ISO-3166 alpha-2, e.g. 'US').

    Same hard rule as everything else on this path: a number is only ever shown when a human
    has set `verified: true` for that block. The file ships entirely unverified, so this is
    inert until someone checks it. CRISIS lines are preferred over the general emergency
    number — 911 reaches police, which is not what a person in despair needs first.
    """
    import os
    code = os.environ.get("SATSANG_COUNTRY", "").strip().upper()
    if not code:
        return ""
    cfg = Path(__file__).resolve().parent.parent / "config" / "emergency_numbers.yaml"
    data = _load_helpline_config(cfg)
    lines = []
    crisis = (data.get("crisis_by_country") or {}).get(code)
    if crisis and crisis.get("verified") and crisis.get("lines"):
        lines += list(crisis["lines"])
    # EU states mostly have no national entry here, and their emergency number reaches
    # police — not a counsellor. The EU-wide 116 xxx lines ARE the emotional-support
    # route, so surface them for member states that lack their own crisis block.
    eu = data.get("eu_wide") or {}
    if code in _EU27 and eu.get("verified") and eu.get("lines"):
        lines += [ln for ln in eu["lines"] if not ln.startswith("112")]
    if data.get("verified"):                      # general emergency number, if confirmed
        num = (data.get("emergency_by_country") or {}).get(code)
        if num:
            lines.append(f"Emergency services: {num}")
    # Dedupe on the DIGITS, not the text: a national entry and the EU-wide entry can be
    # the same number under different labels (Ireland: "Samaritans 116 123" and
    # "116 123 — emotional support"). Repeating a number in a crisis message looks careless.
    seen: set[str] = set()
    deduped: list[str] = []
    for ln in lines:
        key = re.sub(r"\D", "", ln)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(ln)
    lines = deduped
    if not lines:
        return ""
    return f"\nIn your country ({code}):\n" + "\n".join(f"• {ln}" for ln in lines)


# Static, human-reviewed responses. Keep warm + brief; lead with care, give the line.
_DIRECTORY = ("\nIf you are outside India, please find a local crisis line at "
              "findahelpline.com or befrienders.org, or call your local emergency number."
              + _regional_appendix())

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
    # Method-seeking gets its own words: it must decline the information explicitly —
    # warmly, without lecturing — and still lead with care rather than refusal.
    "suicide_method": ("I'm not going to help with that, and I hope you'll forgive me for "
                       "saying it plainly — because I'd rather you were still here to be "
                       "annoyed with me. Something is very heavy right now for you to be "
                       "asking it. Please let someone stay with you through tonight.\n\n"
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
    # suicide_method first: "how many pills…" carries immediate risk and its phrasing
    # often matches nothing else, so it must not be shadowed by a later category.
    for cat in ("suicide_method", "suicide", "self_harm", "violence", "abuse"):
        if any(p.search(text) for p in _COMPILED[cat]):
            return CrisisResult(True, cat, _RESPONSES[cat])
    return CrisisResult(False)
