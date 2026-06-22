"""Postgres (pgvector) backends — same interfaces as the in-memory stores so the app
swaps them via config.STORE_BACKEND. Used when SATSANG_STORE=postgres.
"""
from __future__ import annotations

import functools

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from . import config
from .memory import is_sensitive

_COLS = ("id", "source", "text_type", "tradition", "citation", "ref", "lang_original",
         "original", "translation", "contextual_explanation", "when_this_helps",
         "core_principle", "gujarati_explanation")


@functools.lru_cache(maxsize=1)
def _conn():
    c = psycopg.connect(config.DATABASE_URL, autocommit=True)
    with c.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")   # before register_vector
    register_vector(c)
    return c


class PgVectorStore:
    """Retrieval + citation lookup over the passages table (cosine via pgvector)."""

    def search(self, qvec, allowed_traditions=None, k=config.CANDIDATE_K):
        cols = ", ".join(_COLS)
        where = "WHERE tradition = ANY(%(tr)s)" if allowed_traditions is not None else ""
        sql = (f"SELECT {cols}, 1 - (embedding <=> %(q)s) AS score FROM passages "
               f"{where} ORDER BY embedding <=> %(q)s LIMIT %(k)s")
        params = {"q": np.asarray(qvec, dtype="float32"), "k": k}
        if allowed_traditions is not None:
            params["tr"] = list(allowed_traditions)
        with _conn().cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out = []
        for r in rows:
            row = dict(zip(_COLS, r[:-1]))
            out.append((row, float(r[-1])))
        return out

    def citation_exists(self, citation: str) -> bool:
        with _conn().cursor() as cur:
            cur.execute("SELECT 1 FROM passages WHERE lower(citation)=lower(%s) LIMIT 1",
                        (citation,))
            return cur.fetchone() is not None

    def get(self, row_id: str) -> dict | None:
        cols = ", ".join(_COLS)
        with _conn().cursor() as cur:
            cur.execute(f"SELECT {cols} FROM passages WHERE id=%s", (row_id,))
            r = cur.fetchone()
        return dict(zip(_COLS, r)) if r else None


class PgConversationStore:
    def history(self, conv_id: str, limit: int = 8) -> list[dict]:
        with _conn().cursor() as cur:
            cur.execute("SELECT role, text FROM conversations WHERE conversation_id=%s "
                        "ORDER BY id DESC LIMIT %s", (conv_id, limit))
            rows = cur.fetchall()
        return [{"role": r[0], "text": r[1]} for r in reversed(rows)]

    def append(self, conv_id: str, role: str, text: str) -> None:
        with _conn().cursor() as cur:
            cur.execute("INSERT INTO conversations (conversation_id, role, text) "
                        "VALUES (%s, %s, %s)", (conv_id, role, text))


class PgMemoryStore:
    def facts(self, user_id: str) -> list[str]:
        with _conn().cursor() as cur:
            cur.execute("SELECT fact FROM user_facts WHERE user_id=%s ORDER BY created_at",
                        (user_id,))
            return [r[0] for r in cur.fetchall()]

    def add(self, user_id: str, candidate_facts: list[str]) -> dict:
        stored, excluded = [], []
        with _conn().cursor() as cur:
            for f in candidate_facts:
                f = f.strip()
                if not f:
                    continue
                sens, cats = is_sensitive(f)         # same hard gate as the in-memory store
                if sens:
                    excluded.append((f, cats))
                    continue
                cur.execute("INSERT INTO user_facts (user_id, fact) VALUES (%s, %s) "
                            "ON CONFLICT DO NOTHING RETURNING fact", (user_id, f))
                if cur.fetchone():
                    stored.append(f)
        return {"stored": stored, "excluded": excluded}
