"""Generate node — the saint's reply. Claude Sonnet 4.6, streaming, grounded ONLY in
the retrieved passages. Cites sources with [P#] tags so verification is deterministic
(every [P#] maps to a real retrieved passage; the model cannot invent a citation).
"""
from __future__ import annotations

import functools

from . import config
from .retrieve import Passage

PERSONA = """You are a warm, patient saint-companion rooted in the Swaminarayan \
(Akshar-Purushottam) tradition and the broader Hindu wisdom. You help ordinary people \
with real, messy life problems.

How you speak:
- Problem-first and human. Meet the person's feeling before any teaching. Be brief and \
warm; never lecture or sermonize.
- Loving, never sycophantic. You may gently push back or challenge — kindly, never harshly.
- Ground every spiritual point in the PASSAGES provided. Do NOT quote verses, names, or \
citations that are not in the passages. If the passages don't fit the person's need, say \
so honestly and offer plain human comfort rather than forcing scripture.
- When you draw on a passage, cite it inline with its tag like [P1]. Only use the tags \
that are given.
- Never give medical, legal, or crisis instructions. Stay within compassion and wisdom.
- Reply in the SAME language/register the person used (English, Hinglish, or Gujarati).
"""


def _passages_block(passages: list[Passage]) -> str:
    out = []
    for i, p in enumerate(passages, 1):
        body = p.translation or p.original
        meaning = p.contextual_explanation
        out.append(f"[P{i}] {p.citation} ({p.source})\n"
                   f"  text: {body[:600]}\n  meaning: {meaning[:400]}")
    return "\n\n".join(out) if out else "(no passages retrieved)"


@functools.lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic()


def stream_reply(message: str, plan: dict, passages: list[Passage]):
    """Yield response text chunks. plan is the understand() dict."""
    user = (f"The person wrote:\n\"{message}\"\n\n"
            f"Their underlying problem: {plan.get('problem_summary','')}\n"
            f"Felt emotion: {plan.get('primary_emotion','')}\n"
            f"How to help: {plan.get('response_plan','')}\n\n"
            f"PASSAGES (cite only these, by tag):\n{_passages_block(passages)}\n\n"
            f"Respond to the person now as the saint-companion.")
    with _client().messages.stream(
        model=config.GEN_MODEL, max_tokens=1024,
        system=[{"type": "text", "text": PERSONA, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    ) as stream:
        for text in stream.text_stream:
            yield text
