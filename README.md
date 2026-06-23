# SatsangAI — AI + Application Layer

A warm **"saint" companion** that helps people with real, messy life problems through the
wisdom of Hindu + Swaminarayan sacred texts — **problem-first, zero-hallucination,
multilingual**. This repo is the AI + application layer; the knowledge base is a separate,
finished project (`../satsangai`) that this repo consumes.

> **Status (2026-06): V1 backend complete, rigorously evaluated, and deployable.** The KB is
> enriched and live on HuggingFace; the FastAPI app + chat UI run end-to-end on CPU + Claude API.

---

## What it does

```
safety classifier (deterministic, first; cannot be bypassed)
  → understand + plan        (1 Claude Sonnet 4.6 JSON call: emotion, mode, search queries)
  → retrieve                 (BGE-M3 query embed → tradition-filtered search over the
                              enriched counseling core → cross-encoder rerank; no LLM)
  → generate                 (Claude Sonnet 4.6, streaming, saint persona, grounded ONLY
                              in retrieved passages, [P#] citation tags)
  → verify citations         (deterministic: every [P#] maps to a real passage; no LLM)
  → memory                   (short-term: full history; long-term: per-user facts with a
                              HARD sensitive-data exclusion gate)
```

**Core principles:** problem-first (understand the human before scripture); zero
hallucination (citations verified by DB lookup, never an LLM); saint persona (warm,
patient, lovingly pushes back, never sycophantic); safety-first (deterministic crisis gate
with verified helplines); tradition-aware (home = Swaminarayan/BAPS; never mix schools in
counseling); memory with hard sensitive-data exclusion.

---

## Repo structure

| Path | What |
|---|---|
| `config/counseling_core.yaml` | The curated 17,808-row "counseling core" (retrieval default + enrichment scope) |
| `enrichment/` | The KB enrichment pipeline (DONE): gold seed → QLoRA-tuned Gemma 4 → enrich → embed → write back → push to HF |
| `api/` | The V1 FastAPI backend (pipeline nodes, stores, web UI) |
| `api/web/index.html` | Self-contained chat UI (no build step) |
| `api/db/`, `api/pg.py`, `api/store.py` | Postgres (pgvector) backend, swappable via `SATSANG_STORE` |
| `api/tests/` | Deterministic safety + memory-gate tests |
| `eval/` | Evaluation harness (Opus-judge): response quality, drift, hallucination, topic-switch, retrieval-lift |
| `docker-compose.yml` | pgvector Postgres for the `postgres` store |

The **knowledge base** (231,940 rows, BGE-M3 embeddings, enrichment written back) lives in
`../satsangai` and on HuggingFace `aarsh-adhvaryu/satsangai-kb` (private).

---

## Quickstart

```bash
# 0. deps already installed in this Studio: transformers, sentence-transformers, anthropic,
#    fastapi/uvicorn, psycopg[binary], pgvector
export ANTHROPIC_API_KEY=...                 # lives in ~/.zshrc here: `source ~/.zshrc`

# 1. build the retrieval index from the enriched KB (once; ~seconds)
python -m api.build_index

# 2. run (in-memory stores — default)
HF_HUB_OFFLINE=1 uvicorn api.main:app --host 0.0.0.0 --port 8000
#    → open http://localhost:8000/  (chat UI),  POST /chat (SSE),  GET /health
```

### Postgres backend (production)
```bash
docker compose up -d
SATSANG_DATABASE_URL=postgresql://postgres:satsang@localhost:5433/satsang python -m api.db.load_pg
SATSANG_STORE=postgres SATSANG_DATABASE_URL=... uvicorn api.main:app --port 8000
```

### Config toggles (env)
`SATSANG_STORE` (memory|postgres) · `SATSANG_RERANK=0` (off) · `SATSANG_FAITHFULNESS_GUARD=1`
(non-streaming; revises unfaithful claims) · `SATSANG_EMBED_DEVICE=cuda` · `SATSANG_HELPLINES_VERIFIED`.

---

## Evaluation

```bash
python -m api.tests.test_safety_memory                  # deterministic safety + memory gates
HF_HUB_OFFLINE=1 python -m eval.run_eval                # response quality (Opus judge)
HF_HUB_OFFLINE=1 python -m eval.topic_switch            # topic changes within a session
HF_HUB_OFFLINE=1 python -m eval.retrieval_lift          # enriched vs old embeddings
```

| Dimension | Result |
|---|---|
| Drift / boundaries (not a therapist) | **5/5** (refuses diagnosis/prescription/therapy/legal) |
| Crisis gate (deterministic) | **2/2** |
| Hallucination / fabrication | **2/2** (no fake verses, no fabricated citations) |
| Topic-switch in one session | **5/5** |
| Response faithfulness / persona | **14/15** (residual ~7% is strict-judge noise on subtle paraphrase) |
| Safety + memory unit tests | **all pass** (sensitive data never persisted) |
| Retrieval-lift (enriched vs old) | tie — *honest finding: BGE-M3 cross-lingual strength means enrichment's value is in generation grounding, not retrieval vectors* |

---

## Safety & privacy

- **Crisis gate** (`api/safety.py`) is deterministic, runs first, biased to over-trigger;
  returns a static, human-reviewed response with **verified India-core helplines**
  (Tele-MANAS 14416, KIRAN, Vandrevala, Women 181, NCW, Childline 1098, 112) + a global
  directory. Regional/diaspora lines still to be added.
- **Long-term memory** never stores self-harm / abuse / trauma / medical / criminal
  disclosures — a deterministic gate backstops the LLM fact-extractor (verified in tests
  and on the Postgres backend).

---

## Status & roadmap

**Done:** KB enrichment (17,804/17,808, on HF) · V1 pipeline · rerank · multi-turn + memory ·
faithfulness-check node · crisis safety (verified helplines) · chat UI · Postgres backend ·
full eval suite.

**Next:** app **Dockerfile** for one-command deploy · regional/diaspora helplines · frontend
polish · **V2** — QLoRA + DPO generation model (Gemma 4, Claude-free, fed by V1 conversation
data in Postgres).

See `CLAUDE.md` for the detailed engineering notes, gotchas, and exact resume state.
