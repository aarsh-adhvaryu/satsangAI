"""Query embedding with BGE-M3 — the SAME model the KB was embedded with, so query
and corpus vectors live in the same space (proposal correction: not Voyage/OpenAI).

CPU is fine for single queries; set SATSANG_EMBED_DEVICE=cuda for throughput.
"""
from __future__ import annotations

import functools
import os

import numpy as np

from . import config


@functools.lru_cache(maxsize=1)
def _model():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")    # BGE-M3 is cached locally
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)


def embed_query(text: str) -> np.ndarray:
    """Return a 1024-d unit-norm float32 vector."""
    v = _model().encode([text], normalize_embeddings=True)[0]
    return np.asarray(v, dtype="float32")
