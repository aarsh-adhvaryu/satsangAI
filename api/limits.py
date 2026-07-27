"""Rate limits and a concurrency guard for a single-GPU deployment.

Sized from MEASURED latency on this hardware, not guesswork (api/data/traces.jsonl,
warm Gemma backend, H100):

    understand  8.3s      retrieve 0.2s      generate 29.6s      total ~38s / turn

One model instance, no continuous batching, so turns are effectively serialised: the
ceiling is ~95 turns/hour. That single number sets everything below. Two simultaneous
generations do not go twice as fast — they make both slower and risk two threads each
loading 52 GB, which is the bug that once hung an eval at zero completions.

The limits are deliberately generous for a person and tight for a script: a thoughtful
human writes every 1-3 minutes, so a 20-second floor is invisible to them and fatal to
rapid-fire. Nothing here needs Redis at this scale — in-memory windows are correct for a
single process, and a restart clearing them is an acceptable trade.
"""
from __future__ import annotations

import collections
import threading
import time

MAX_MESSAGE_CHARS = 4_000        # longest real message so far was ~350; 10x headroom
MIN_INTERVAL_S = 20              # per user, between messages
PER_HOUR = 30                    # ~19 min of GPU: a deep conversation, not a monopoly
PER_DAY = 100
MAX_CONCURRENT = 1               # one model instance
MAX_QUEUED = 4                   # 5th caller would wait 2.5 min — refuse kindly instead
GENERATE_TIMEOUT_S = 120         # median generate 29.6s, max 35.3s

_lock = threading.Lock()
_hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
_sema = threading.BoundedSemaphore(MAX_CONCURRENT)
_queued = 0


class Rejected(Exception):
    """Refusal with a message meant for a person, not a status code."""

    def __init__(self, message: str, retry_after: int = 20):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


def check_message(text: str) -> None:
    if len(text or "") > MAX_MESSAGE_CHARS:
        raise Rejected(
            f"That is longer than I can take in at once ({len(text):,} characters). "
            f"Could you share the heart of it in a few paragraphs?", retry_after=0)


def check_rate(user_id: str) -> None:
    """Sliding windows per user. Messages are worded as the companion, not as an error —
    someone reaching out should never be met with 'HTTP 429'."""
    now = time.time()
    with _lock:
        q = _hits[user_id]
        while q and now - q[0] > 86_400:
            q.popleft()
        last = q[-1] if q else 0
        hour = sum(1 for t in q if now - t <= 3_600)
        if now - last < MIN_INTERVAL_S:
            raise Rejected("Let me finish thinking about the last one — try again in a few "
                           "seconds.", retry_after=int(MIN_INTERVAL_S - (now - last)) + 1)
        if hour >= PER_HOUR:
            raise Rejected("We have talked a great deal this hour. Let it settle, and come "
                           "back to me a little later.", retry_after=600)
        if len(q) >= PER_DAY:
            raise Rejected("That is a lot for one day. Rest, and find me again tomorrow.",
                           retry_after=3_600)
        q.append(now)


class slot:
    """Serialise generation. Refuses rather than queueing deeply, because a spinner that
    never resolves reads as broken and is worse than an honest 'I am with someone else'."""

    def __enter__(self):
        global _queued
        with _lock:
            if _queued >= MAX_QUEUED:
                raise Rejected("I am with someone else just now. Give me a minute and try "
                               "again.", retry_after=60)
            _queued += 1
        try:
            self._acquired = _sema.acquire(timeout=180)
        finally:
            with _lock:
                _queued -= 1
        if not self._acquired:
            raise Rejected("I could not get to you in time — please try once more.",
                           retry_after=30)
        return self

    def __exit__(self, *exc):
        if getattr(self, "_acquired", False):
            _sema.release()
        return False


def status() -> dict:
    with _lock:
        return {"in_flight": MAX_CONCURRENT - _sema._value, "queued": _queued,
                "tracked_users": len(_hits),
                "limits": {"max_message_chars": MAX_MESSAGE_CHARS,
                           "min_interval_s": MIN_INTERVAL_S, "per_hour": PER_HOUR,
                           "per_day": PER_DAY, "max_concurrent": MAX_CONCURRENT}}
