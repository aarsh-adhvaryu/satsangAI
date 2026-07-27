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

# Storage backend: "memory" (parquet + JSON files; default) or "postgres" (pgvector).
STORE_BACKEND = os.environ.get("SATSANG_STORE", "memory")
DATABASE_URL = os.environ.get(
    "SATSANG_DATABASE_URL", "postgresql://postgres:satsang@localhost:5433/satsang")

# Models
EMBED_MODEL = os.environ.get("SATSANG_EMBED_MODEL", "BAAI/bge-m3")   # 1024-d, unit-norm
RERANK_MODEL = os.environ.get("SATSANG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")  # multilingual
RERANK = os.environ.get("SATSANG_RERANK", "1") != "0"               # cross-encoder rerank on
# Faithfulness guard: generate -> check claims -> revise once if unfaithful (before send).
# Stronger zero-hallucination, but non-streaming + 1-2 extra LLM calls. Off by default.
FAITHFULNESS_GUARD = os.environ.get("SATSANG_FAITHFULNESS_GUARD", "0") == "1"

# Crisis helplines in api/safety.py: India-core set human-verified 2026-06. Regional/
# Gujarat-specific and additional diaspora lines still to be added later. Set to "0" to
# re-enable the startup warning if the lines are edited and need re-verification.
CRISIS_HELPLINES_VERIFIED = os.environ.get("SATSANG_HELPLINES_VERIFIED", "1") == "1"
GEN_MODEL = os.environ.get("SATSANG_GEN_MODEL", "claude-sonnet-4-6")  # saint generation
PLAN_MODEL = os.environ.get("SATSANG_PLAN_MODEL", "claude-sonnet-4-6")  # understand+plan JSON
EMBED_DEVICE = os.environ.get("SATSANG_EMBED_DEVICE", "cpu")          # cpu fine for single queries

# Generation backend: "claude" (Sonnet, default) or "gemma" (the from-scratch V2
# adapter — needs a GPU, loads ~52 GB bf16). This switches the saint's REPLY only.
GEN_BACKEND = os.environ.get("SATSANG_GEN_BACKEND", "claude")

# Shastrarth is an OPT-IN mode, off by default and NEVER auto-routed.
# It fails two of its six gates (hallucination 0.85, scripture 0.77) because the
# acharya-school rows it retrieves are unenriched raw OCR. Rather than let the router
# drop sincere learners into the weakest path, the user selects it deliberately — and
# with this flag off it is not offered at all. Turn on with SATSANG_SHASTRARTH=1 once
# the school rows are enriched and the gates pass.
SHASTRARTH_ENABLED = os.environ.get("SATSANG_SHASTRARTH", "0") == "1"

# Crisis-helpline routing. Declared here with every other knob so a deployer has ONE place
# to look; api/safety.py reads the env directly at call time so tests can vary them.
COUNTRY = os.environ.get("SATSANG_COUNTRY", "").strip().upper()   # ISO-3166 alpha-2
REGION = os.environ.get("SATSANG_REGION", "").strip().lower()     # e.g. "gujarat"

# Utility backend: understand+plan and memory fact-extraction. Setting BOTH this and
# GEN_BACKEND to "gemma" is what makes the runtime genuinely Claude-free — with only
# GEN_BACKEND switched, every turn still made two Anthropic calls and the system could
# not start without a key. Proposal §10 specifies Gemma 4 E4B for this role: fast,
# no fine-tuning needed, far cheaper than routing planning through the 26B.
UTILITY_BACKEND = os.environ.get("SATSANG_UTILITY_BACKEND", "claude")
# Empty = reuse the generation model already in VRAM with its adapter disabled (no extra
# load, no download). §10 nominates Gemma 4 E4B for this role; set this once that model id
# is confirmed and cached. The 26B base is what is actually on disk today.
UTILITY_MODEL = os.environ.get("SATSANG_UTILITY_MODEL", "")
UNDERSTAND_MODEL = PLAN_MODEL          # alias used by api/llm.py when routing to Claude
# Default V2 adapter = the DPO2 (on-policy) run. Blinded Opus pairwise judge vs the
# SFT adapter: overall 5-4-3 for DPO2, and loving_pushback (anti-sycophancy — the
# whole point of DPO) 4-0 in its favour, never worse; small faithfulness cost (1-3)
# that api/verify catches deterministically regardless. Set the env var to the
# gemma4-v2-sft-lora dir to serve the SFT adapter instead.
GEMMA_ADAPTER = os.environ.get(
    "SATSANG_GEMMA_ADAPTER", str(ROOT / "v2" / "data" / "gemma4-v2-dpo2-lora"))
GEMMA_MAX_NEW_TOKENS = int(os.environ.get("SATSANG_GEMMA_MAX_NEW_TOKENS", "512"))

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
