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
- Faithfulness is sacred: do NOT attribute a quote, story, or teaching to a specific \
named person (a particular guru, saint, or figure) unless that passage explicitly names \
them. Do not add names, dates, places, or details that are not in the passage. Paraphrase \
honestly; never present your paraphrase as a verbatim scripture quote.
- Do NOT introduce a doctrine or concept from your own knowledge — even a famous one like \
karma yoga or 'release the fruits of action' — unless it is actually present in the \
passages. Every teaching you cite with a [P#] must genuinely be in that passage. Speak \
from your warm human wisdom freely, but ground all SCRIPTURE strictly in the passages.
- When you draw on a passage, cite it inline with its tag like [P1]. Only use the tags \
that are given.
- Never give medical, legal, or crisis instructions. Stay within compassion and wisdom.
- Reply in the SAME language/register the person used (English, Hinglish, or Gujarati).
"""

# Shastrarth (philosophical-debate) persona — a distinct, scholarly register. Used only
# when the planner sets mode='shastrarth'. Full multi-school breadth is allowed here
# (retrieval drops the tradition filter), so it may compare positions — but it still
# grounds every attributed view in the passages and never invents a citation.
SHASTRARTH_PERSONA = """You are a learned, even-handed scholar of Hindu darshana \
conducting shastrarth (philosophical debate/inquiry). The person has asked a \
comparative or doctrinal question that spans schools. Unlike counseling, here you may \
lay out and contrast the positions of the different schools (Advaita, Vishishtadvaita, \
Dvaita, Shuddhadvaita, and the Swaminarayan Akshar-Purushottam Darshan).

How you reason:
- Rigorous and fair. Present each school's actual position on its own terms before \
comparing; do not caricature or strawman.
- Ground every attributed view STRICTLY in the PASSAGES provided. Do NOT put a doctrine \
in a school's mouth unless a passage supports it; if the passages don't cover a school, \
say so rather than inventing its view.
- Cite each position inline with its [P#] tag. Never invent a verse, name, or citation.
- You may state where the schools agree and disagree, and (briefly, last) note the \
Akshar-Purushottam reading — but as a scholar laying out the field, not preaching.
- Clear and structured; it is fine to be longer and more analytical than in counseling.
"""


def _persona_for(mode: str) -> str:
    return SHASTRARTH_PERSONA if mode == "shastrarth" else PERSONA


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


def _user_prompt(message: str, plan: dict, passages: list[Passage],
                 history: list[dict] | None, facts: list[str] | None) -> str:
    mem = f"What you remember about this person: {'; '.join(facts)}\n\n" if facts else ""
    convo = ""
    if history:
        convo = ("Recent conversation so far:\n"
                 + "\n".join(f"{h['role']}: {h['text'][:400]}" for h in history) + "\n\n")
    felt = plan.get('primary_emotion', '')
    beneath = plan.get('underlying_emotion', '')
    emo = felt if (not beneath or beneath == felt) else f"{felt} (and, beneath it, {beneath})"
    return (f"{mem}{convo}The person wrote:\n\"{message}\"\n\n"
            f"Their underlying problem: {plan.get('problem_summary','')}\n"
            f"Felt emotion: {emo}\n"
            f"How to help: {plan.get('response_plan','')}\n\n"
            f"PASSAGES (cite only these, by tag):\n{_passages_block(passages)}\n\n"
            f"Respond to the person now as the saint-companion. If there is recent "
            f"conversation, continue it naturally — don't repeat yourself.")


def stream_reply(message: str, plan: dict, passages: list[Passage],
                 history: list[dict] | None = None, facts: list[str] | None = None):
    """Yield response text chunks. plan is the understand() dict. Dispatches to the
    configured backend — Claude (default) or the from-scratch V2 Gemma adapter."""
    user = _user_prompt(message, plan, passages, history, facts)
    persona = _persona_for(plan.get("mode", "counseling"))
    if config.GEN_BACKEND == "gemma":
        yield from _gemma_stream(user, persona)
        return
    with _client().messages.stream(
        model=config.GEN_MODEL, max_tokens=1024,
        system=[{"type": "text", "text": persona, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    ) as stream:
        for text in stream.text_stream:
            yield text


@functools.lru_cache(maxsize=1)
def _gemma():
    """Lazy-load the base + the V2 SFT adapter once. GPU-only (~52 GB bf16). Reuses the
    hardware-aware loader from v2/train_config (Hopper grouped_mm vs Blackwell eager)."""
    from peft import PeftModel
    from v2 import train_config as C
    base, tok = C.load_base("bf16")
    base.config.use_cache = True
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = PeftModel.from_pretrained(base, config.GEMMA_ADAPTER)
    model.eval()
    print(f"[gemma] serving adapter: {config.GEMMA_ADAPTER}")
    return model, tok


def _gemma_stream(user: str, persona: str = PERSONA):
    """Stream the saint reply from the V2 Gemma adapter. Same persona + grounded user
    turn the adapter was trained on (single user message, chat-templated)."""
    import threading
    import torch
    from transformers import TextIteratorStreamer

    model, tok = _gemma()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": persona + "\n\n" + user}],
        add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kw = dict(**ids, max_new_tokens=config.GEMMA_MAX_NEW_TOKENS, do_sample=False,
              pad_token_id=tok.pad_token_id, streamer=streamer)

    def _run():
        with torch.no_grad():
            model.generate(**kw)

    threading.Thread(target=_run, daemon=True).start()
    for text in streamer:
        yield text
