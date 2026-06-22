"""Understand + plan — ONE Claude call (structured JSON). Analyzes the human problem,
names the emotion, decides retrieval mode, and writes the search queries the retriever
will embed. Problem-first: understand the person before reaching for scripture.
"""
from __future__ import annotations

import functools
import json

from . import config

SYSTEM = (
    "You triage messages for a warm 'saint' companion that helps people with real life "
    "problems using Hindu and Swaminarayan scripture. You do NOT reply to the user. You "
    "analyze their message and produce retrieval queries for a problem-first search.\n"
    "- primary_emotion: the dominant feeling in plain words.\n"
    "- problem_summary: one neutral sentence naming the real human problem.\n"
    "- mode: 'counseling' for personal/emotional/practical problems (default); "
    "'shastrarth' ONLY if they explicitly ask a comparative philosophical/doctrinal "
    "question across schools.\n"
    "- search_queries: 2-4 short queries describing the underlying need/theme (e.g. "
    "'letting go of anger toward family', 'finding steadiness in loss') — NOT keywords, "
    "and NOT scripture names.\n"
    "- response_plan: one sentence on how to help (tone + what to address)."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "primary_emotion": {"type": "string"},
        "problem_summary": {"type": "string"},
        "mode": {"type": "string", "enum": ["counseling", "shastrarth"]},
        "search_queries": {"type": "array", "items": {"type": "string"}},
        "response_plan": {"type": "string"},
    },
    "required": ["primary_emotion", "problem_summary", "mode", "search_queries",
                 "response_plan"],
    "additionalProperties": False,
}


@functools.lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic()


def understand(message: str, history: list[dict] | None = None) -> dict:
    ctx = ""
    if history:
        ctx = ("Recent conversation (for context):\n"
               + "\n".join(f"{h['role']}: {h['text'][:300]}" for h in history)
               + "\n\n")
    content = f"{ctx}Latest message to analyze:\n{message}"
    resp = _client().messages.create(
        model=config.PLAN_MODEL, max_tokens=600,
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    data = json.loads(text)
    if not data.get("search_queries"):
        data["search_queries"] = [message]      # fallback: search the raw message
    return data
