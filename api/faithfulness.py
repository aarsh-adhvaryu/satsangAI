"""Claim-level faithfulness check (+ optional one-shot revision) — closes the residual
where a warm paraphrase slightly exceeds what the passages actually support.

- check(): lists any claim in the reply not supported by the passages.
- guarded_generate(): generate -> check -> if unfaithful, revise ONCE to ground/remove
  the flagged claims -> re-check. Trades streaming + a little latency/cost for stronger
  zero-hallucination guarantees (the product's core value). Gate with SATSANG_FAITHFULNESS_GUARD.
"""
from __future__ import annotations

import functools
import json

from . import config
from .generate import _client, _passages_block, stream_reply, PERSONA
from .retrieve_types import Passage

_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["faithful", "unsupported_claims"],
    "additionalProperties": False,
}

_CHECK_SYSTEM = (
    "You verify faithfulness for a scripture-grounded companion. Given the PASSAGES the "
    "system was given and its RESPONSE, list every claim in the response that is NOT "
    "supported by the passages — invented doctrines, quotes/teachings not present, "
    "speaker/name/detail attributions the passages don't make, or famous concepts pulled "
    "from outside the passages. Plain human empathy and advice are fine and are NOT "
    "claims. Only flag scriptural/doctrinal/attribution claims. Return strict JSON."
)


def check(reply: str, passages: list[Passage]) -> dict:
    content = f"PASSAGES:\n{_passages_block(passages)}\n\nRESPONSE:\n{reply}"
    r = _client().messages.create(
        model=config.GEN_MODEL, max_tokens=500,
        system=[{"type": "text", "text": _CHECK_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": _CHECK_SCHEMA}})
    return json.loads(next(b.text for b in r.content if b.type == "text"))


def _revise(message: str, passages: list[Passage], reply: str, unsupported: list[str]) -> str:
    fix = "\n".join(f"- {c}" for c in unsupported)
    user = (f"Your previous reply to the person contained claims NOT supported by the "
            f"passages:\n{fix}\n\nRewrite your reply so every scriptural/attribution claim "
            f"is grounded in the passages below — remove or soften the unsupported ones, "
            f"keep the same warmth and the valid [P#] citations, and do not introduce new "
            f"ungrounded claims.\n\nPASSAGES:\n{_passages_block(passages)}\n\n"
            f"The person originally wrote:\n\"{message}\"\n\nYour previous reply:\n{reply}")
    r = _client().messages.create(
        model=config.GEN_MODEL, max_tokens=1024,
        system=[{"type": "text", "text": PERSONA, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}])
    return next((b.text for b in r.content if b.type == "text"), reply)


def guarded_generate(message: str, plan: dict, passages: list[Passage],
                     history=None, facts=None) -> tuple[str, dict]:
    """Returns (final_reply, report). report = {faithful, unsupported_claims, revised}."""
    reply = "".join(stream_reply(message, plan, passages, history=history, facts=facts))
    chk = check(reply, passages)
    if chk["faithful"] or not chk["unsupported_claims"]:
        return reply, {**chk, "revised": False}
    revised = _revise(message, passages, reply, chk["unsupported_claims"])
    chk2 = check(revised, passages)
    return revised, {**chk2, "revised": True}
