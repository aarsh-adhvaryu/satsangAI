"""Understand + plan — ONE Claude call (structured JSON). Analyzes the human problem,
names the emotion, decides retrieval mode, and writes the search queries the retriever
will embed. Problem-first: understand the person before reaching for scripture.
"""
from __future__ import annotations

import json

from . import config

SYSTEM = (
    "You triage messages for a warm 'saint' companion that helps people with real life "
    "problems using Hindu and Swaminarayan scripture. You do NOT reply to the user. You "
    "analyze their message and produce retrieval queries for a problem-first search.\n"
    "- primary_emotion: the SURFACE feeling the person expresses, in plain words.\n"
    "- underlying_emotion: the feeling that may sit BENEATH the surface one and drive it "
    "(e.g. anger masking hurt; certainty masking fear; humor masking loneliness). If the "
    "surface feeling is clearly the whole story, repeat it. One or two plain words.\n"
    "- problem_summary: one neutral sentence naming the real human problem.\n"
    "- mode: 'counseling' for personal/emotional/practical problems (DEFAULT — if the "
    "person is hurting, this wins even when they phrase it as a question); "
    "'teaching' when they want to UNDERSTAND something about the tradition rather than "
    "work through a problem — what a concept means, what the tradition holds, how a "
    "practice works, or an explanation of a specific verse (e.g. 'do Hindus believe in "
    "reincarnation?', 'what is the difference between atma and jiva?', 'is it wrong to "
    "eat meat?'); 'creative' when they ask you to WRITE something original for them (a poem, prayer, kirtan, satsang talk or reflection); 'shastrarth' ONLY when they explicitly want the SCHOOLS COMPARED "
    "against each other (Advaita vs Vishishtadvaita vs Dvaita) or ask for a scholarly "
    "debate. A sincere learner asking a general doctrinal question is 'teaching', NOT "
    "'shastrarth' — shastrarth is for someone who already knows the schools.\n"
    "- search_queries: 2-4 short queries describing the underlying need/theme (e.g. "
    "'letting go of anger toward family', 'finding steadiness in loss') — NOT keywords, "
    "and NOT scripture names.\n"
    "- in_domain: TRUE for anything this companion should answer — personal or emotional "
    "problems, relationships, work/money stress as a human struggle, meaning and purpose, "
    "faith and doubt, scripture, doctrine, practice, festivals, ethics, a request to write "
    "something devotional, or plain human conversation. FALSE ONLY when the person wants "
    "technical, professional or factual help unrelated to spiritual life or their inner "
    "world — code, maths homework, medical or legal advice, business/tax mechanics, sports, "
    "product recommendations, current news. When in doubt answer TRUE: a person circling a "
    "hard thing often opens with a strange question, and wrongly turning them away is far "
    "worse than answering warmly.\n"
    "- response_plan: one sentence on how to help (tone + what to address)."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "primary_emotion": {"type": "string"},
        "underlying_emotion": {"type": "string"},
        "problem_summary": {"type": "string"},
        "mode": {"type": "string", "enum": ["counseling", "teaching", "creative", "shastrarth"]},
        "search_queries": {"type": "array", "items": {"type": "string"}},
        "in_domain": {"type": "boolean"},
        "response_plan": {"type": "string"},
    },
    "required": ["primary_emotion", "underlying_emotion", "problem_summary", "mode",
                 "in_domain", "search_queries", "response_plan"],
    "additionalProperties": False,
}


def understand(message: str, history: list[dict] | None = None,
               allow_shastrarth: bool = False) -> dict:
    """Plan the turn. Routed through api/llm so it runs on Claude OR Gemma 4 — with
    SATSANG_UTILITY_BACKEND=gemma this node stops calling Anthropic entirely.

    `allow_shastrarth` is False unless the USER explicitly selected that mode. The
    router never picks it on its own: it is the one mode whose retrieval reaches the
    unenriched school corpus, so an auto-route would silently hand a learner the
    weakest path. When disallowed it is removed from the schema entirely, so the model
    cannot emit it, and a stray value is downgraded to 'teaching' below.
    """
    from .llm import complete_json
    schema = SCHEMA
    system = SYSTEM
    if not allow_shastrarth:
        schema = json.loads(json.dumps(SCHEMA))          # deep copy; don't mutate module state
        schema["properties"]["mode"]["enum"] = ["counseling", "teaching", "creative"]
        system = SYSTEM + ("\nNOTE: shastrarth mode is unavailable. Comparative questions "
                           "about the schools should be answered in 'teaching' mode, grounded "
                           "in the home tradition's own sources.")
    ctx = ""
    if history:
        ctx = ("Recent conversation (for context):\n"
               + "\n".join(f"{h['role']}: {h['text'][:300]}" for h in history)
               + "\n\n")
    content = f"{ctx}Latest message to analyze:\n{message}"
    # A planning failure must degrade, never crash the turn: searching the raw message
    # in counseling mode is a safe default that still retrieves something sensible.
    fallback = {"mode": "counseling", "search_queries": [message], "in_domain": True}
    data = complete_json(system, content, schema=schema, model=config.PLAN_MODEL,
                         max_tokens=600, fallback=fallback)
    if not data.get("search_queries"):
        data["search_queries"] = [message]      # fallback: search the raw message
    if not data.get("mode"):
        data["mode"] = "counseling"
    if data["mode"] == "shastrarth" and not allow_shastrarth:
        data["mode"] = "teaching"               # belt-and-braces: schema already excluded it
    if data["mode"] not in ("counseling", "teaching", "creative", "shastrarth"):
        data["mode"] = "counseling"
    # Default to answering. A planner that fails to emit the field must not silently turn
    # someone away, and the cost of wrongly declining a real struggle is much higher than
    # the cost of answering an off-topic question warmly.
    data["in_domain"] = bool(data.get("in_domain", True))
    return data
