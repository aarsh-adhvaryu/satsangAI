"""V1 request pipeline — orchestrates the nodes in order:

    safety (deterministic, first) -> understand+plan (1 Claude call)
    -> retrieve (BGE-M3 + tradition filter; no LLM)
    -> generate (Claude Sonnet 4.6, streaming, grounded)
    -> verify citations (deterministic; no LLM)

Yields (event_type, payload) tuples so the API can stream them as SSE.
"""
from __future__ import annotations

from . import safety
from .generate import stream_reply
from .retrieve import retrieve
from .understand import understand
from .verify import verify


def respond(message: str):
    # 1. Safety gate — runs first, cannot be bypassed.
    crisis = safety.classify(message)
    if crisis.is_crisis:
        yield "crisis", {"category": crisis.category}
        yield "text", crisis.response
        yield "done", {"crisis": True, "cited": [], "unverified_refs": []}
        return

    # 2. Understand + plan (1 Claude JSON call).
    plan = understand(message)
    yield "plan", plan

    # 3. Retrieve (no LLM).
    passages = retrieve(plan["search_queries"], mode=plan.get("mode", "counseling"),
                        rerank_query=plan.get("problem_summary") or message)
    yield "passages", [{"tag": f"[P{i}]", "citation": p.citation, "source": p.source,
                        "tradition": p.tradition, "score": round(p.score, 3)}
                       for i, p in enumerate(passages, 1)]

    # 4. Generate (Claude, streaming, grounded only in passages).
    full = []
    for chunk in stream_reply(message, plan, passages):
        full.append(chunk)
        yield "text", chunk

    # 5. Verify citations (deterministic).
    yield "done", verify("".join(full), passages)
