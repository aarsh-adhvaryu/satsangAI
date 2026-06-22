"""Storage factories — return the in-memory or Postgres backend per config.STORE_BACKEND.
The app calls these instead of constructing stores directly, so swapping is one env var.
"""
from __future__ import annotations

import functools

from . import config


@functools.lru_cache(maxsize=1)
def vector_store():
    if config.STORE_BACKEND == "postgres":
        from .pg import PgVectorStore
        return PgVectorStore()
    from .index import get_index
    return get_index()                 # Index already exposes search/citation_exists/get


@functools.lru_cache(maxsize=1)
def conversation_store():
    if config.STORE_BACKEND == "postgres":
        from .pg import PgConversationStore
        return PgConversationStore()
    from .memory import ConversationStore
    return ConversationStore()


@functools.lru_cache(maxsize=1)
def fact_store():
    if config.STORE_BACKEND == "postgres":
        from .pg import PgMemoryStore
        return PgMemoryStore()
    from .memory import MemoryStore
    return MemoryStore()
