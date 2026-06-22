"""V1 backend config — paths, models, retrieval policy.

Corrections from the proposal are applied here (see ../CLAUDE.md):
- query embeddings use BGE-M3 (same as the KB), never Voyage/OpenAI;
- default retrieval = the enriched counseling core, not the full 231k corpus;
- generation = Claude Sonnet 4.6.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # satsangAI/
KB = ROOT.parent / "satsangai" / "data" / "parquet"    # the built KB
KB_CORPUS = KB / "corpus.parquet"
KB_F32 = KB / "embeddings.f32"
KB_META = KB / "embeddings_meta.json"

INDEX_PATH = ROOT / "api" / "data" / "counseling_index.parquet"

# Models
EMBED_MODEL = os.environ.get("SATSANG_EMBED_MODEL", "BAAI/bge-m3")   # 1024-d, unit-norm
RERANK_MODEL = os.environ.get("SATSANG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")  # multilingual
RERANK = os.environ.get("SATSANG_RERANK", "1") != "0"               # cross-encoder rerank on
# Faithfulness guard: generate -> check claims -> revise once if unfaithful (before send).
# Stronger zero-hallucination, but non-streaming + 1-2 extra LLM calls. Off by default.
FAITHFULNESS_GUARD = os.environ.get("SATSANG_FAITHFULNESS_GUARD", "0") == "1"

# Crisis helplines in api/safety.py are PLACEHOLDERS. A human MUST verify every number
# (and add region-appropriate lines) and set SATSANG_HELPLINES_VERIFIED=1 before any
# real/production use. The app warns loudly at startup while this is False.
CRISIS_HELPLINES_VERIFIED = os.environ.get("SATSANG_HELPLINES_VERIFIED", "0") == "1"
GEN_MODEL = os.environ.get("SATSANG_GEN_MODEL", "claude-sonnet-4-6")  # saint generation
PLAN_MODEL = os.environ.get("SATSANG_PLAN_MODEL", "claude-sonnet-4-6")  # understand+plan JSON
EMBED_DEVICE = os.environ.get("SATSANG_EMBED_DEVICE", "cpu")          # cpu fine for single queries

# Traditions
HOME_TRADITION = "swaminarayan"                  # Akshar-Purushottam / BAPS
SHARED = "shared_hindu"
# The acharya philosophical schools — full breadth ONLY in Shastrarth mode; never
# mixed into counseling (proposal: "never mix schools in counseling").
SCHOOLS = ("advaita", "vishishtadvaita", "dvaita", "shuddhadvaita")

# Counseling retrieval defaults: home tradition + shared Hindu, widen if thin.
COUNSELING_TRADITIONS = (HOME_TRADITION, SHARED)

# Retrieval params
CANDIDATE_K = 40        # vector recall before rerank
TOP_K = 6               # passages handed to the generator
MIN_SCORE = 0.35        # cosine floor; below this we consider results "thin"
