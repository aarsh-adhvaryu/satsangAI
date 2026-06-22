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
import json
import re
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


# --- LLM fact extraction (gated by is_sensitive on write) ----------------------
EXTRACT_SYSTEM = (
    "Extract durable, NON-sensitive facts about the user worth remembering across "
    "future conversations (name, location, family/relationship structure in neutral "
    "terms, language preference, ongoing goals or interests). "
    "DO NOT extract anything about self-harm, abuse, trauma, grief/loss, medical or "
    "mental-health conditions, or legal/criminal matters. If there is nothing durable "
    "and safe to remember, return an empty list. Return STRICT JSON: {\"facts\": [..]}"
)


@functools.lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic()


def extract_facts(message: str, reply: str) -> list[str]:
    msg = f"User said:\n{message}\n\nAssistant replied:\n{reply}\n\nExtract durable safe facts."
    resp = _client().messages.create(
        model=MEMORY_MODEL, max_tokens=300,
        system=[{"type": "text", "text": EXTRACT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": msg}],
        output_config={"format": {"type": "json_schema", "schema": {
            "type": "object", "properties": {"facts": {"type": "array",
            "items": {"type": "string"}}}, "required": ["facts"],
            "additionalProperties": False}}})
    text = next((b.text for b in resp.content if b.type == "text"), '{"facts":[]}')
    return json.loads(text).get("facts", [])
