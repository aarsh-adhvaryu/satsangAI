"""Run the evaluation: deterministic checks + LLM judge over the labeled cases.

Per case:
- crisis cases: assert the deterministic gate fired with a static helpline response and
  NO LLM/retrieval ran (safety is the hard guarantee).
- everything else: run the full pipeline, then check
    * deterministic citation validity (verify.all_ok — no hallucinated citations)
    * Opus judge: faithful / no_fabrication / respects_boundaries / in_persona

Writes a per-case table + per-category pass rates.

    source ~/.zshrc && HF_HUB_OFFLINE=1 python -m eval.run_eval
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

from api import safety
from api.generate import _passages_block, stream_reply
from api.retrieve import retrieve
from api.understand import understand
from api.verify import verify
from .cases import CASES
from .judge import judge


def _run_case(case: dict) -> dict:
    msg = case["message"]
    # crisis is deterministic — verify the gate WITHOUT invoking the LLM pipeline
    crisis = safety.classify(msg)
    if case["category"] == "crisis":
        ok = crisis.is_crisis and bool(crisis.response) and "helpline" in crisis.response.lower()
        return {"id": case["id"], "category": case["category"], "crisis_handled": ok,
                "pass": ok, "detail": f"category={crisis.category}"}

    # non-crisis: a crisis flag here would be a false positive (still report it)
    if crisis.is_crisis:
        return {"id": case["id"], "category": case["category"], "pass": False,
                "detail": f"UNEXPECTED crisis flag ({crisis.category})"}

    # run the nodes directly so the judge sees EXACTLY the passages the generator saw
    from api import config
    plan = understand(msg)
    passages = retrieve(plan["search_queries"], mode=plan.get("mode", "counseling"),
                        rerank_query=plan.get("problem_summary") or msg)
    if config.FAITHFULNESS_GUARD:
        from api.faithfulness import guarded_generate
        reply, _ = guarded_generate(msg, plan, passages)
    else:
        reply = "".join(stream_reply(msg, plan, passages))
    verify_result = verify(reply, passages)
    block = _passages_block(passages)

    j = judge(msg, block, reply)
    citations_ok = verify_result.get("all_ok", False)
    overall = j["overall_pass"] and citations_ok
    return {"id": case["id"], "category": case["category"], "pass": overall,
            "citations_ok": citations_ok, **{k: j[k] for k in
            ("faithful", "no_fabrication", "respects_boundaries", "in_persona")},
            "detail": j["rationale"][:160],
            "unsupported": j["unsupported_claims"][:3]}


def _run_case_resilient(case: dict, retries: int = 4) -> dict:
    """Retry transient API errors (529/overload/rate-limit) with backoff; never abort."""
    import anthropic
    for attempt in range(retries):
        try:
            return _run_case(case)
        except (anthropic.OverloadedError, anthropic.RateLimitError,
                anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            wait = 8 * (2 ** attempt)
            print(f"  [{case['id']}] transient {type(e).__name__}; retry in {wait}s")
            time.sleep(wait)
        except Exception as e:  # hard failure — record and move on
            return {"id": case["id"], "category": case["category"], "pass": False,
                    "detail": f"ERROR {type(e).__name__}: {str(e)[:100]}"}
    return {"id": case["id"], "category": case["category"], "pass": False,
            "detail": "ERROR: exhausted retries"}


def main() -> None:
    results = [_run_case_resilient(c) for c in CASES]
    by_cat = defaultdict(lambda: [0, 0])
    print(f"\n{'id':16} {'cat':14} {'PASS':5} detail")
    print("-" * 90)
    for r in results:
        by_cat[r["category"]][0] += int(r["pass"])
        by_cat[r["category"]][1] += 1
        print(f"{r['id']:16} {r['category']:14} {'✅' if r['pass'] else '❌':5} {r.get('detail','')}")
        if not r["pass"] and r.get("unsupported"):
            print(f"{'':37}unsupported: {r['unsupported']}")

    print("\n=== pass rate by category ===")
    for cat, (p, n) in sorted(by_cat.items()):
        print(f"  {cat:14} {p}/{n}")
    total_p = sum(p for p, _ in by_cat.values())
    total_n = sum(n for _, n in by_cat.values())
    print(f"  {'TOTAL':14} {total_p}/{total_n}")
    json.dump(results, open("eval/last_results.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
