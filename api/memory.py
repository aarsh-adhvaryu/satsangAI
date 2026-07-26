"""Memory — short-term conversation history + long-term per-user facts, with a HARD
sensitive-data exclusion rule (proposal): self-harm / abuse / trauma / medical /
criminal disclosures are NEVER written to long-term memory.

Design: short-term history keeps EVERYTHING (needed for a coherent conversation);
long-term memory is a small set of durable, NON-sensitive facts. The deterministic
`is_sensitive` gate is the backstop — even if the LLM extractor proposes a sensitive
fact, the gate drops it. File-backed JSON for V1 (Postgres in production).
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import time
from pathlib import Path

from . import config

MEM_DIR = config.ROOT / "api" / "data" / "memory"
MEMORY_MODEL = "claude-haiku-4-5"           # cheap; fact extraction is a small task

# --- Deterministic sensitivity gate (long-term-memory exclusion) ---------------
# Deliberately broad; over-exclusion is privacy-safe, under-exclusion is the harm.
_SENSITIVE: dict[str, list[str]] = {
    "self_harm": [r"\bsuicid", r"\bkill(ing)?\s+myself\b", r"\bself[-\s]?harm",
                  r"\bend(ing)?\s+(my|this)\s+life\b", r"\bcut(ting)?\s+myself\b",
                  r"\bwant\s+to\s+die\b"],
    "abuse": [r"\babus(e|ed|ive)\b", r"\b(hit|beat|molest|rape|assault)(s|ed|ing)?\b",
              r"\bdomestic\s+violence\b"],
    "trauma": [r"\btrauma", r"\bptsd\b", r"\bpanic\s+attack", r"\bflashback",
               r"\bnightmare", r"\bgrie(f|ving)\b", r"\bbereave", r"\bpassed\s+away\b",
               r"\bdied\b", r"\bdeath\s+of\b", r"\bmiscarriage"],
    "medical": [r"\bdiagnos(ed|is)\b", r"\bcancer\b", r"\bdepress(ion|ed)\b",
                r"\banxiety\s+disorder", r"\bbipolar\b", r"\bschizo", r"\bmedication\b",
                r"\bmedicine\b", r"\bantidepressant", r"\btherap(y|ist)\b", r"\bpsychiatr",
                r"\billness\b", r"\bdisease\b", r"\bhiv\b", r"\bpregnan", r"\bsurgery\b",
                r"\bhospital", r"\baddict"],
    "criminal": [r"\barrest", r"\bjail\b", r"\bprison\b", r"\bpolice\b", r"\bstole\b",
                 r"\bstealing\b", r"\bfraud\b", r"\billegal\b", r"\bcrime\b", r"\bconvicted\b"],
}
_SENS_C = {k: [re.compile(p, re.I) for p in v] for k, v in _SENSITIVE.items()}


def is_sensitive(text: str) -> tuple[bool, list[str]]:
    cats = [c for c, pats in _SENS_C.items() if any(p.search(text) for p in pats)]
    return bool(cats), cats


# --- Stores --------------------------------------------------------------------
def _load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


class ConversationStore:
    """Short-term: full turn history per conversation (keeps everything)."""
    def __init__(self):
        self.path = MEM_DIR / "conversations.json"

    def history(self, conv_id: str, limit: int = 8) -> list[dict]:
        return _load(self.path, {}).get(conv_id, [])[-limit:]

    def append(self, conv_id: str, role: str, text: str) -> None:
        data = _load(self.path, {})
        data.setdefault(conv_id, []).append({"role": role, "text": text})
        _save(self.path, data)


class MemoryStore:
    """Long-term: durable, NON-sensitive facts per user. Sensitive facts are dropped."""
    def __init__(self):
        self.path = MEM_DIR / "memory.json"

    def facts(self, user_id: str) -> list[str]:
        return _load(self.path, {}).get(user_id, [])

    @staticmethod
    def _norm(f: str) -> str:
        return re.sub(r"^(the\s+)?user'?s?\s+", "", f.strip().lower()).strip(" .")

    def add(self, user_id: str, candidate_facts: list[str]) -> dict:
        """Returns {stored: [...], excluded: [(fact, cats)]} — gate is the backstop."""
        data = _load(self.path, {})
        existing = data.setdefault(user_id, [])
        seen = {self._norm(x) for x in existing}
        stored, excluded = [], []
        for f in candidate_facts:
            f = f.strip()
            if not f:
                continue
            sens, cats = is_sensitive(f)
            if sens:
                excluded.append((f, cats))            # HARD-excluded, never persisted
            elif self._norm(f) not in seen:
                existing.append(f)
                seen.add(self._norm(f))
                stored.append(f)
        _save(self.path, data)
        return {"stored": stored, "excluded": excluded}

    # --- §7 "control is absolute": view / edit / delete / clear -----------------
    # Facts are stored as bare strings, so identity is derived from the normalised
    # text rather than a stored key. That keeps every existing memory.json readable
    # with no migration, and an edit is simply delete-then-add.
    @classmethod
    def fact_id(cls, fact: str) -> str:
        return hashlib.sha1(cls._norm(fact).encode()).hexdigest()[:12]

    def items(self, user_id: str) -> list[dict]:
        """Facts with stable ids, for a memory panel the user can act on."""
        return [{"id": self.fact_id(f), "text": f} for f in self.facts(user_id)]

    def update(self, user_id: str, fact_id: str, new_text: str) -> dict:
        """Edit one fact. The sensitivity gate applies to user edits too — a person
        must not be able to hand-write a self-harm disclosure into durable storage."""
        new_text = new_text.strip()
        if not new_text:
            return {"ok": False, "error": "empty"}
        sens, cats = is_sensitive(new_text)
        if sens:
            return {"ok": False, "error": "sensitive", "categories": cats}
        data = _load(self.path, {})
        facts = data.get(user_id, [])
        for i, f in enumerate(facts):
            if self.fact_id(f) == fact_id:
                facts[i] = new_text
                _save(self.path, data)
                return {"ok": True, "id": self.fact_id(new_text), "text": new_text}
        return {"ok": False, "error": "not_found"}

    def delete(self, user_id: str, fact_id: str) -> dict:
        data = _load(self.path, {})
        facts = data.get(user_id, [])
        kept = [f for f in facts if self.fact_id(f) != fact_id]
        if len(kept) == len(facts):
            return {"ok": False, "error": "not_found"}
        data[user_id] = kept
        _save(self.path, data)
        return {"ok": True, "deleted": fact_id, "remaining": len(kept)}

    def clear(self, user_id: str) -> dict:
        data = _load(self.path, {})
        n = len(data.get(user_id, []))
        data[user_id] = []
        _save(self.path, data)
        return {"ok": True, "cleared": n}


class PrefsStore:
    """Per-user controls (§7) + interaction memory.

    `paused` stops long-term fact extraction entirely without deleting what's stored;
    `consent` gates whether this user's conversations may be retained for training
    (proposal §29). Both default to OFF for consent and OFF for paused — i.e. memory
    works, but nothing is training-eligible until the person opts in.
    """
    DEFAULTS = {"paused": False, "consent": False,
                "language": None, "length": None, "style": None}

    def __init__(self):
        self.path = MEM_DIR / "prefs.json"

    def get(self, user_id: str) -> dict:
        return {**self.DEFAULTS, **_load(self.path, {}).get(user_id, {})}

    def set(self, user_id: str, **changes) -> dict:
        data = _load(self.path, {})
        cur = {**self.DEFAULTS, **data.get(user_id, {})}
        for k, v in changes.items():
            if k in self.DEFAULTS and v is not None:
                cur[k] = v
        data[user_id] = cur
        _save(self.path, data)
        return cur


class FeedbackStore:
    """Turn-level ratings — the signal that turns served conversations into DPO pairs.

    Without this a deployed conversation leaves no trace of whether the reply was any
    good, so it cannot become a preference pair (proposal §8/§29). Stored append-only
    with the reply text so a pair can be reconstructed later.
    """
    def __init__(self):
        self.path = MEM_DIR / "feedback.jsonl"

    def add(self, *, user_id: str | None, conversation_id: str | None, rating: str,
            message: str = "", reply: str = "", note: str = "") -> dict:
        if rating not in ("up", "down"):
            return {"ok": False, "error": "rating must be 'up' or 'down'"}
        row = {"ts": time.time(), "user_id": user_id, "conversation_id": conversation_id,
               "rating": rating, "message": message, "reply": reply, "note": note}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"ok": True}

    def all(self, user_id: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if user_id is None or r.get("user_id") == user_id:
                out.append(r)
        return out


def export_user(user_id: str, conversations=None, facts=None,
                prefs=None, feedback=None) -> dict:
    """Everything held about one person, in one JSON payload (§7 'export or clear').

    Stores are passed in so this works against either backend (JSON or Postgres).
    Conversation history is included because it is data about the user even though it
    is short-term; sensitive disclosures were never written to `facts` by design.
    """
    facts = facts if facts is not None else MemoryStore()
    prefs = prefs if prefs is not None else PrefsStore()
    feedback = feedback if feedback is not None else FeedbackStore()
    convos = {}
    if conversations is not None:
        raw = _load(getattr(conversations, "path", MEM_DIR / "conversations.json"), {})
        convos = raw if isinstance(raw, dict) else {}
    return {
        "user_id": user_id,
        "exported_at": time.time(),
        "facts": facts.items(user_id) if hasattr(facts, "items") else facts.facts(user_id),
        "preferences": prefs.get(user_id),
        "feedback": feedback.all(user_id),
        "conversations": convos,
        "note": ("Sensitive disclosures (self-harm, abuse, trauma, medical, criminal) are "
                 "never written to long-term memory — see is_sensitive()."),
    }


# --- LLM fact extraction (gated by is_sensitive on write) ----------------------
EXTRACT_SYSTEM = (
    "Extract durable, NON-sensitive facts about the user worth remembering across "
    "future conversations (name, location, family/relationship structure in neutral "
    "terms, language preference, ongoing goals or interests). "
    "DO NOT extract anything about self-harm, abuse, trauma, grief/loss, medical or "
    "mental-health conditions, or legal/criminal matters. If there is nothing durable "
    "and safe to remember, return an empty list. Return STRICT JSON: {\"facts\": [..]}"
)


_FACTS_SCHEMA = {"type": "object",
                 "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
                 "required": ["facts"], "additionalProperties": False}


def extract_facts(message: str, reply: str) -> list[str]:
    """Propose durable facts. Routed through api/llm so it runs on Claude OR Gemma 4.

    Whatever the backend proposes, `MemoryStore.add` re-checks every candidate against
    the deterministic `is_sensitive` gate before anything is written — so swapping in a
    smaller utility model cannot weaken the privacy guarantee.
    """
    from .llm import complete_json
    msg = f"User said:\n{message}\n\nAssistant replied:\n{reply}\n\nExtract durable safe facts."
    data = complete_json(EXTRACT_SYSTEM, msg, schema=_FACTS_SCHEMA, model=MEMORY_MODEL,
                         max_tokens=300, fallback={"facts": []})
    facts = data.get("facts") or []
    return [f for f in facts if isinstance(f, str)]
