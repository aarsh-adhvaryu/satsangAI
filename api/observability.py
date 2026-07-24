"""Lightweight request tracing / observability (proposal §21).

Vendor-free by default: per-request spans (stage timings), token/citation counts, and a
crisis/mode/backend tag, appended to a JSONL trace log and kept in a ring buffer for a
`/metrics` endpoint. If LANGSMITH_API_KEY is set and `langsmith` is installed, each trace
is also shipped there — but nothing here depends on it.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from pathlib import Path

_TRACE_LOG = Path(__file__).resolve().parent / "data" / "traces.jsonl"
_RING: deque = deque(maxlen=200)          # recent traces for /metrics


class Trace:
    """Times named spans within one request. Use as: t=Trace(); with t.span('retrieve'): ..."""

    def __init__(self, **tags):
        self.id = "req_" + uuid.uuid4().hex[:12]
        self.t0 = time.perf_counter()
        self.spans: dict[str, float] = {}
        self.tags = tags
        self._stack: list[tuple[str, float]] = []

    def span(self, name: str):
        trace = self

        class _Ctx:
            def __enter__(self):
                self._s = time.perf_counter()
                return self

            def __exit__(self, *exc):
                trace.spans[name] = round((time.perf_counter() - self._s) * 1000, 1)
                return False
        return _Ctx()

    def set(self, **tags):
        self.tags.update({k: v for k, v in tags.items() if v is not None})

    def finish(self) -> dict:
        rec = {"id": self.id, "total_ms": round((time.perf_counter() - self.t0) * 1000, 1),
               "spans_ms": self.spans, **self.tags}
        _RING.append(rec)
        try:
            _TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with _TRACE_LOG.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        if os.environ.get("LANGSMITH_API_KEY"):
            _ship_langsmith(rec)
        return rec


def recent(n: int = 50) -> list[dict]:
    return list(_RING)[-n:]


def summary() -> dict:
    """Aggregate stats over the ring buffer — for /metrics."""
    rs = list(_RING)
    if not rs:
        return {"requests": 0}
    tot = sorted(r["total_ms"] for r in rs)
    stage_avg: dict[str, float] = {}
    for r in rs:
        for k, v in r.get("spans_ms", {}).items():
            stage_avg[k] = stage_avg.get(k, 0.0) + v
    stage_avg = {k: round(v / len(rs), 1) for k, v in stage_avg.items()}
    return {
        "requests": len(rs),
        "latency_ms": {"p50": tot[len(tot) // 2], "p95": tot[int(len(tot) * 0.95)], "max": tot[-1]},
        "avg_stage_ms": stage_avg,
        "crisis_rate": round(sum(1 for r in rs if r.get("crisis")) / len(rs), 3),
        "backend": rs[-1].get("backend"),
    }


def _ship_langsmith(rec: dict) -> None:
    try:
        from langsmith import Client
        Client().create_run(name="satsang_request", run_type="chain",
                             inputs={"tags": rec}, outputs={}, extra={"metadata": rec})
    except Exception:
        pass
