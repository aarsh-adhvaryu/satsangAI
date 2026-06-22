"""V1 request pipeline — orchestrates the nodes in order:

    safety (deterministic, first) -> understand+plan (1 Claude call, context-aware)
    -> retrieve (BGE-M3 + tradition filter + rerank; no LLM)
    -> generate (Claude Sonnet 4.6, streaming, grounded, memory+history aware)
    -> verify citations (deterministic; no LLM)
    -> update memory (short-term history always; long-term facts gated)

Conversation/user state is optional — omit ids for stateless single-turn.
Yields (event_type, payload) tuples so the API can stream them as SSE.
"""
from __future__ import annotations

from . import config, safety
from .generate import stream_reply
from .memory import extract_facts
from .retrieve import retrieve
from .store import conversation_store, fact_store
from .understand import understand
from .verify import verify

_convos = conversation_store()
_memory = fact_store()


def respond(message: str, conversation_id: str | None = None, user_id: str | None = None):
    history = _convos.history(conversation_id) if conversation_id else []
    facts = _memory.facts(user_id) if user_id else []

    # 1. Safety gate — runs first, cannot be bypassed.
    crisis = safety.classify(message)
    if crisis.is_crisis:
        if conversation_id:
            _convos.append(conversation_id, "user", message)
            _convos.append(conversation_id, "assistant", crisis.response)
        yield "crisis", {"category": crisis.category}
        yield "text", crisis.response
        yield "done", {"crisis": True, "cited": [], "unverified_refs": []}
        return  # never extract memory from a crisis turn

    # 2. Understand + plan (context-aware).
    plan = understand(message, history=history)
    yield "plan", plan

    # 3. Retrieve (no LLM).
    passages = retrieve(plan["search_queries"], mode=plan.get("mode", "counseling"),
                        rerank_query=plan.get("problem_summary") or message)
    yield "passages", [{"tag": f"[P{i}]", "citation": p.citation, "source": p.source,
                        "tradition": p.tradition,
                        "score": round(p.rerank_score if p.rerank_score is not None else p.score, 3)}
                       for i, p in enumerate(passages, 1)]

    # 4. Generate (grounded, memory + history aware).
    if config.FAITHFULNESS_GUARD:
        from .faithfulness import guarded_generate
        reply, faith = guarded_generate(message, plan, passages, history=history, facts=facts)
        yield "text", reply                     # guarded mode is non-streaming
    else:
        full = []
        for chunk in stream_reply(message, plan, passages, history=history, facts=facts):
            full.append(chunk)
            yield "text", chunk
        reply = "".join(full)
        faith = None

    # 5. Verify citations (deterministic) + faithfulness report.
    result = verify(reply, passages)
    if faith is not None:
        result["faithfulness"] = faith

    # 6. Update memory: short-term always; long-term facts gated by is_sensitive.
    if conversation_id:
        _convos.append(conversation_id, "user", message)
        _convos.append(conversation_id, "assistant", reply)
    if user_id:
        mem = _memory.add(user_id, extract_facts(message, reply))
        result["memory"] = {"stored": mem["stored"],
                            "excluded": [{"fact": f, "categories": c} for f, c in mem["excluded"]]}
    yield "done", result
