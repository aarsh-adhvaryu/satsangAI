"""`chosen`-side generation for the DPO bootstrap (OFFLINE only).

The Hybrid decision: bootstrap the preferred replies with Claude offline (same
persona + grounding as V1 serves), gated by the deterministic citation verifier so
we never teach DPO an ungrounded `chosen`. This is offline gold — runtime stays
Gemma-only, and it's reversible (source untouched), exactly like the enrichment gold.

Claude-free alternative (`--chosen gemma`) self-generates from the SFT'd model; it's
stubbed here until an initial SFT adapter exists (chicken-and-egg: the first model
must be bootstrapped).
"""
from __future__ import annotations

from api.generate import PERSONA
from api.retrieve_types import Passage
from api.verify import verify
from v2.schema import context_from_passages

GEN_MODEL = "claude-sonnet-4-6"          # the model V1 serves — chosen matches V1 quality
NATURALIZE_MODEL = "claude-haiku-4-5"    # cheap paraphrase task

_NAT_SYS = (
    "You rewrite a terse, clinical description of someone's struggle into a natural, "
    "first-person message that a real person would actually type to a warm spiritual "
    "companion. Keep the emotional content and situation faithful. Make it conversational "
    "and specific — 1 to 3 sentences. Do NOT add scripture, advice, or details that "
    "weren't implied. Vary the phrasing and opening. Output ONLY the rewritten message.")


def naturalize_problem(problem: str, client, model: str = NATURALIZE_MODEL) -> str:
    """Turn a regex-mangled seed problem into a natural user message. Falls back to the
    original on any failure so the pipeline never stalls."""
    try:
        resp = client.messages.create(
            model=model, max_tokens=200, system=_NAT_SYS,
            messages=[{"role": "user", "content": problem}])
        txt = "".join(b.text for b in resp.content
                      if getattr(b, "type", "") == "text").strip()
        return txt or problem
    except Exception:
        return problem


def build_user_turn(problem: str, passages: list[Passage]) -> str:
    return (f"The person wrote:\n\"{problem}\"\n\n"
            f"PASSAGES (cite only these, by tag):\n{context_from_passages(passages)}\n\n"
            f"Respond to the person now as the saint-companion. Keep it warm and brief; "
            f"cite passages you draw on with their [P#] tag.")


def is_grounded(reply: str, passages: list[Passage]) -> bool:
    """Quality gate: no fabricated references (every [P#] maps to a real passage)."""
    return not verify(reply, passages).get("unverified_refs")


def generate_claude(problem: str, passages: list[Passage], client, model: str = GEN_MODEL,
                    max_retries: int = 2) -> str | None:
    """A faithful, grounded saint reply; None if it can't pass the verifier."""
    for _ in range(max_retries + 1):
        resp = client.messages.create(
            model=model, max_tokens=700, system=PERSONA,
            messages=[{"role": "user", "content": build_user_turn(problem, passages)}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if text and is_grounded(text, passages):
            return text
    return None                          # dropped rather than poison the pair set
