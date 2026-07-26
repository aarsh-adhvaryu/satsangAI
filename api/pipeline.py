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

from . import config, creative, safety, verse
from .generate import stream_reply
from .memory import PrefsStore, extract_facts
from .observability import Trace
from .retrieve import retrieve
from .store import conversation_store, fact_store
from .understand import understand
from .verify import verify

_convos = conversation_store()
_memory = fact_store()
_prefs = PrefsStore()


def prepare(message: str, history: list[dict] | None = None, mode: str | None = None,
            tr=None) -> tuple[dict, list]:
    """Everything between the safety gate and generation: plan, domain gate, verse lookup,
    creative detection, retrieval. Returns (plan, passages).

    Extracted so `respond()` and the evaluation harness share ONE implementation. They did
    not: `eval/six_gate._live_reply` had its own copy of understand->retrieve->generate and
    silently missed the domain gate and verse lookup, so a whole eval run scored
    out_of_domain 0/12 and verse 0/15 on ROUTING while the product itself was correct.
    Any new routing step belongs here, and both callers get it.
    """
    from contextlib import nullcontext
    span = tr.span if tr is not None else (lambda _name: nullcontext())
    wants_shastrarth = (mode == "shastrarth") and config.SHASTRARTH_ENABLED

    with span("understand"):
        plan = understand(message, history=history, allow_shastrarth=wants_shastrarth)
    if wants_shastrarth:
        plan["mode"] = "shastrarth"          # explicit user selection wins over the router

    # Domain gate: out-of-domain answers are ungrounded by definition, so nothing is
    # retrieved and the persona gives an honest refusal instead of a confident guess.
    if not plan.get("in_domain", True):
        plan["mode"] = "out_of_domain"
        return plan, []

    # §5.2 verse lookup — DETERMINISTIC, not left to the router.
    ref = verse.parse_reference(message)
    verse_row = verse.lookup(ref) if ref else None
    if verse_row is not None:
        plan["mode"] = "verse"
        plan["verse_reference"] = ref
        plan["verse_block"] = verse.render_block(verse.verse_view(verse_row))

    # §5.3/§5.4 creative request (form + output language).
    creative_form = creative.detect_form(message)
    if creative_form:
        plan["mode"] = "creative"
        plan["creative_form"] = creative_form
        plan["creative_language"] = creative.detect_language(message)

    with span("retrieve"):
        passages = retrieve(plan["search_queries"], mode=plan.get("mode", "counseling"),
                            rerank_query=plan.get("problem_summary") or message)
        if verse_row is not None:
            # The requested verse must BE [P1]: otherwise the model narrates a verse the
            # verifier has no passage for and every claim about it reads as uncited.
            from .retrieve_types import Passage
            pinned = Passage.from_row(verse_row, score=1.0)
            passages = [pinned] + [p for p in passages if p.id != pinned.id]

    if creative_form:
        # Built after retrieval: the attribution contract must name the passages actually
        # available to quote and to credit.
        if plan.get("creative_language"):
            plan["creative_instruction"] = creative.creative_instruction(
                creative_form, plan["creative_language"], passages)
        else:
            plan["creative_instruction"] = (
                f"They asked for a {creative_form}, but have NOT said which language they "
                f"want it in. Do NOT write the piece yet. Reply with one short, warm "
                f"sentence asking: {creative.ASK_LANGUAGE} Nothing else.")
    return plan, passages


def respond(message: str, conversation_id: str | None = None, user_id: str | None = None,
            mode: str | None = None):
    """`mode` is the user's explicit selection from the client (a mode picker, like the
    model selectors in other chat UIs). It exists so shastrarth is only ever entered
    deliberately — see config.SHASTRARTH_ENABLED. Selecting it while the feature flag is
    off has no effect; every other mode is decided by the router as before."""
    wants_shastrarth = (mode == "shastrarth") and config.SHASTRARTH_ENABLED
    history = _convos.history(conversation_id) if conversation_id else []
    facts = _memory.facts(user_id) if user_id else []
    tr = Trace(backend=config.GEN_BACKEND)

    # 1. Safety gate — runs first, cannot be bypassed.
    with tr.span("safety"):
        crisis = safety.classify(message)
    if crisis.is_crisis:
        if conversation_id:
            _convos.append(conversation_id, "user", message)
            _convos.append(conversation_id, "assistant", crisis.response)
        tr.set(crisis=True, category=crisis.category)
        yield "crisis", {"category": crisis.category}
        yield "text", crisis.response
        yield "done", {"crisis": True, "cited": [], "unverified_refs": [], "trace": tr.finish()}
        return  # never extract memory from a crisis turn

    # 2-3. Plan, route (domain gate / verse / creative) and retrieve — shared with the
    # eval harness via prepare(), so the two can never drift apart again.
    plan, passages = prepare(message, history=history, mode=mode, tr=tr)
    tr.set(mode=plan.get("mode"))
    yield "plan", plan

    if plan["mode"] == "out_of_domain":
        yield "passages", []                     # nothing retrieved: nothing to ground in
        decline = []
        with tr.span("generate"):
            for chunk in stream_reply(message, plan, [], history=history, facts=facts):
                decline.append(chunk)
                yield "text", chunk
        reply = "".join(decline)
        if conversation_id:
            _convos.append(conversation_id, "user", message)
            _convos.append(conversation_id, "assistant", reply)
        yield "done", {"out_of_domain": True, "cited": [], "unverified_refs": [],
                       "all_ok": True, "trace": tr.finish()}
        return

    yield "passages", [{"tag": f"[P{i}]", "citation": p.citation, "source": p.source,
                        "tradition": p.tradition,
                        "score": round(p.rerank_score if p.rerank_score is not None else p.score, 3)}
                       for i, p in enumerate(passages, 1)]

    # 4. Generate (grounded, memory + history aware).
    with tr.span("generate"):
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
    with tr.span("verify"):
        result = verify(reply, passages)
        # §19: creative output carries a stricter contract — a line tagged as scripture
        # must genuinely MATCH the passage (no adjusting a verse to fit the metre), and
        # original writing must be attributed. Only enforced once a piece was actually
        # written; the language question is not a creative piece.
        if plan.get("creative_form") and plan.get("creative_language"):
            result["attribution"] = creative.verify_creative(reply, passages)
            result["all_ok"] = result["all_ok"] and result["attribution"]["all_ok"]
    result["trace"] = tr.finish()
    if faith is not None:
        result["faithfulness"] = faith

    # 6. Update memory: short-term always; long-term facts gated by is_sensitive
    #    AND by the user's own pause switch (§7 — pausing must actually stop writes,
    #    not just hide them, or the control is theatre).
    if conversation_id:
        _convos.append(conversation_id, "user", message)
        _convos.append(conversation_id, "assistant", reply)
    if user_id:
        prefs = _prefs.get(user_id)
        if prefs.get("paused"):
            result["memory"] = {"stored": [], "excluded": [], "paused": True}
        else:
            mem = _memory.add(user_id, extract_facts(message, reply))
            result["memory"] = {
                "stored": mem["stored"],
                "excluded": [{"fact": f, "categories": c} for f, c in mem["excluded"]],
                "paused": False}
        result["consent"] = bool(prefs.get("consent"))   # is this turn training-eligible?
    yield "done", result
