"""Prove the whole product runs with NO Anthropic API — plan, retrieve, generate, verify.

Run:
    env -u ANTHROPIC_API_KEY SATSANG_UTILITY_BACKEND=gemma SATSANG_GEN_BACKEND=gemma \
        SATSANG_EMBED_DEVICE=cuda HF_HUB_OFFLINE=1 python -m eval.local_smoke

One 52 GB base is loaded once and serves both roles: the tuned adapter answers as the
saint, the same base with the adapter disabled does planning and fact extraction. Expect
a few minutes for the initial load, then seconds per turn.
"""
from __future__ import annotations

import os
import time


def main() -> None:
    from api import config
    from api.llm import runtime_is_claude_free
    print(f"ANTHROPIC_API_KEY set: {'ANTHROPIC_API_KEY' in os.environ}")
    print(f"utility={config.UTILITY_BACKEND}  generation={config.GEN_BACKEND}  "
          f"utility_model={config.UTILITY_MODEL or '(reuses generation model)'}")
    print(f"runtime_is_claude_free(): {runtime_is_claude_free()}\n" + "=" * 74)

    from api.pipeline import respond

    CASES = [
        ("counseling",    "I keep losing my temper with my mother and I feel terrible after"),
        ("verse",         "Explain Bhagavad Gita 2.47 to me."),
        ("teaching",      "What's the difference between atma and jiva?"),
        ("out_of_domain", "Explain how photosynthesis works."),
        ("creative",      "Write me a short poem in English about letting go."),
        ("crisis",        "I want to kill myself tonight"),
    ]
    for want, msg in CASES:
        t0 = time.time()
        out, plan, done = [], None, None
        try:
            for ev, payload in respond(msg):
                if ev == "plan":
                    plan = payload
                elif ev == "text":
                    out.append(payload)
                elif ev == "done":
                    done = payload
        except Exception as e:                                   # noqa: BLE001
            print(f"[{want:<13}] FAILED {type(e).__name__}: {str(e)[:110]}")
            continue
        reply = "".join(out).strip()
        mode = (plan or {}).get("mode", "crisis" if (done or {}).get("crisis") else "?")
        cited = len((done or {}).get("cited") or [])
        unver = (done or {}).get("unverified_refs") or []
        print(f"[{want:<13}] mode={mode:<13} {time.time()-t0:5.0f}s  cited={cited} "
              f"unverified={unver}  chars={len(reply)}")
        print("   " + reply[:260].replace("\n", " ") + ("…" if len(reply) > 260 else ""))
        print("-" * 74)


if __name__ == "__main__":
    main()
