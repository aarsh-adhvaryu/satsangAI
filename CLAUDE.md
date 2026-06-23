# SatsangAI — AI + Application Layer

This repo is the **AI + application layer** for SatsangAI: a warm "saint" companion
that helps people with real, messy life problems through the wisdom of Hindu +
Swaminarayan sacred texts — problem-first, zero-hallucination, multilingual.

GitHub: https://github.com/aarsh-adhvaryu/satsangAI

The **knowledge base is a separate, finished project** — do NOT rebuild it here.
This repo *consumes* it and adds enrichment, retrieval, generation, memory, and the API.

## ⏯️ CURRENT STATUS (2026-06) — resume here
**V1 backend is complete, evaluated, and deployable.** Everything below is built, tested,
committed to GitHub `main`, and detailed later in this file. A new session can just continue.
- **KB enrichment DONE + shipped:** 17,804/17,808 counseling-core rows enriched (Gemma 4 26B
  MoE, QLoRA-tuned on Claude-Opus gold, runtime Claude-free), embedded, written back, pushed
  to HF `aarsh-adhvaryu/satsangai-kb` (private). Audited at parity. See "Enrichment pipeline".
- **V1 backend DONE:** `api/` — safety → understand → retrieve(+rerank) → generate → verify →
  memory; chat UI at `/`; Postgres backend swappable via `SATSANG_STORE`. See "V1 backend — run & deploy".
- **Evaluated:** drift/boundaries 5/5, crisis 2/2, hallucination 2/2, topic-switch 5/5, quality
  14/15, safety/memory tests pass; retrieval-lift = honest tie (value is in generation, not vectors).
- **NEXT (not started):** app **Dockerfile** for one-command deploy; regional/diaspora helplines;
  frontend polish; then **V2** = QLoRA + **DPO** generation Gemma (Claude-free), fed by V1's
  Postgres conversation data. Full ongoing state: see project memory `satsangai-project-state`.
- **Run:** `source ~/.zshrc && HF_HUB_OFFLINE=1 uvicorn api.main:app --port 8000` (index built via
  `python -m api.build_index`). Key access: ANTHROPIC_API_KEY in `~/.zshrc` (prefix commands with
  `source ~/.zshrc`). No GPU needed for V1; GPU (Blackwell, eager-MoE) only for V2 enrichment/tuning.

## Core principles (from the proposal)
- **Problem-first, not scripture-first** — understand the human problem, then let
  scripture serve it.
- **Zero hallucination** — every quoted verse is real; citations verified by a
  deterministic DB lookup, never by an LLM.
- **Saint persona** — patient, warm, never lectures; pushes back lovingly (no sycophancy).
- **Safety first** — a deterministic crisis classifier runs BEFORE any LLM and cannot
  be bypassed; static, human-reviewed crisis response with verified helplines.
- **Tradition-aware** — home tradition is Swaminarayan (Akshar-Purushottam / BAPS);
  respectful of the broader Hindu tradition; never mix schools in counseling, full
  breadth only in Shastrarth mode.
- **Memory with hard sensitive-data exclusion** — self-harm/abuse/trauma/medical/
  criminal disclosures are NEVER written to long-term memory.

Full spec: `../.data/data/data/SatsangAI_Final_Complete_Proposal.md`
(Part I = vision, Part II = V2 from-scratch Gemma stack, Part III = V1 Claude+RAG).
Source list: `../.data/data/data/SatsangAI_Final_Ingestion_List.md`.

## The knowledge base (consume, don't recreate)
- HF dataset (PRIVATE): **`aarsh-adhvaryu/satsangai-kb`** — needs an HF token to load.
- **231,940 records**, one row = one chunk. BGE-M3 embeddings (1024-d, unit-norm) on
  every row. KB repo: `../satsangai` (+ https://github.com/aarsh-adhvaryu/satsangAI_KB).
- Per-row schema: `id, source, text_type, tradition` (swaminarayan | shared_hindu |
  advaita | vishishtadvaita | dvaita | shuddhadvaita), `citation, ref, lang_original,
  original, transliteration, translation, word_meanings, commentaries,
  contextual_explanation, when_this_helps, core_principle, gujarati_explanation,
  embedding, embedding_source_text, text_source, ocr_confidence, verified, provenance`.
- **Integrity audited + remediated (2026-06):** mojibake effectively 0; Shikshapatri
  re-OCR'd; Janmangal (109) and Nishkulanand (55) re-scraped; promo-cover junk removed.
  Structurally clean (0 empty required fields, 0 dup ids, OCR gating enforced).

### Critical KB facts that drive this layer's design
- **The enrichment fields are still 100% NULL** (`contextual_explanation`,
  `when_this_helps`, `core_principle`, `gujarati_explanation`, `embedding_source_text`).
  They are the proposal's *primary retrieval target* and do not exist yet.
- **Therefore current embeddings are on `translation` (≈126k rows) or raw `original`**
  (the BGE-M3 fallback), NOT the enrichment layer. Problem-first retrieval is weaker
  than intended until enrichment is generated.
- **Corpus is skewed to narrative** — itihasa 92.7k + Mahabharata 73.7k + Valmiki 21.6k
  dominate; Swaminarayan is only ~14.3k. A naive full-corpus search floods counseling
  with Sanskrit narrative verses.

## Architecture decisions (corrections to the proposal — apply these)
- **Query embeddings MUST use BGE-M3** (same model as the KB), NOT Voyage/OpenAI as
  the proposal's V1 table says — otherwise query and corpus vectors don't align.
- **Default counseling retrieval = a curated "counseling core" index**, not the full
  231k: Vachanamrut, Swamini Vato, Bhagavad Gita, principal Upanishads, Yoga Sutras,
  Shikshapatri, Satsang Diksha, curated `sw_lit_*`. Widen to +shared_hindu when thin;
  drop tradition filters only in Shastrarth mode.
- **Enrichment engine = LOCAL Gemma ONLY, never the Claude API.** The KB is shared
  with the from-scratch V2 (whose whole value is being Claude-free) and embeddings
  derive from enrichment, so Claude-generated enrichment would contaminate V2. The
  `local` backend in `../satsangai/pipeline/enrich.py` is still an unwired stub.
- V1 generation uses **Claude Sonnet 4.6** (`claude-sonnet-4-6`) with prompt caching.

## V1 request pipeline (FastAPI, to build)
```
safety classifier (deterministic, first) → understand+plan (1 Claude JSON call)
→ retrieve (BGE-M3 + tradition filter + Postgres exact lookup + rerank; no LLM)
→ generate (Claude Sonnet 4.6, streaming, grounded only in retrieved passages)
→ verify citations (regex extract + Postgres existence check; no LLM)
```

## Enrichment pipeline — DONE + SHIPPED (2026-06)
**The enrichment layer is complete and live on HuggingFace** (`aarsh-adhvaryu/satsangai-kb`,
private). 17,804 / 17,808 counseling-core rows now carry the 4 enrichment fields +
enrichment-based BGE-M3 embeddings; pushed end-to-end. See "What's done / next / AUDIT" below.

Build order: **V1 (Claude + RAG) first; V2 (Gemma) parallel.** The enrichment layer is
the proposal's primary retrieval target. Decisions locked with the
owner: **quality over speed/cost**; enrichment model = **Gemma 4 26B MoE**
(`google/gemma-4-26B-A4B-it`, Apache-2.0), QLoRA-tuned; tuning gold = Claude (Opus 4.8)
generated **offline only** — runtime enrichment stays Gemma-only so the shared KB stays
runtime-Claude-free for V2 (reversible: source text untouched, re-derivable later).

Code lives in `config/` + `enrichment/` (this repo, the app layer):
- `config/counseling_core.yaml` — the **counseling core** = 17,808 rows (7.7% of the
  231,940 corpus), tiered core / widen-when-thin / shastrarth / excluded-narrative.
  `enrichment/core_filter.py` resolves it against `../satsangai/data/parquet/corpus.parquet`.
- `enrichment/sample_gold_seed.py` → `data/gold_seed_sample.parquet` (1,490 diverse rows,
  sqrt-allocated per source, stratified by text_type×lang).
- `enrichment/prompt.py` — canonical enrichment prompt (mirrors KB `enrich.py`).
- `enrichment/generate_gold.py` — `submit`/`collect`/`retry`. Batch API + cached system
  prompt + structured-output JSON. **DONE: `data/gold.jsonl` = 1,490/1,490 rows, all
  core fields filled, Gujarati on all 1,300 swaminarayan rows.** (~<$20 of API spend.)
- `enrichment/baseline_smoke.py` — un-tuned Gemma baseline (already faithful + multilingual).
- `enrichment/qlora_train.py` — 4-bit bnb + LoRA r16 on the 210 language_model attn+MLP
  `Linear4bit` targets, completion-only loss. **DONE: 3 epochs, train_loss → ~1.0; adapter
  at `data/gemma4-enrich-lora/`** (gitignored, on the Studio disk).
- `enrichment/enrich_core.py` — bulk enrichment (merges LoRA, batched, eager MoE, resumable;
  flags `--batch`, `--priority`, `--retry-bad`, `--max-new-tokens`). **DONE: 17,804/17,808
  rows in `data/enriched_core.jsonl`.** Tip: `--batch 48` ~tripled throughput.
- `enrichment/embed_core.py` — BGE-M3 embed on `contextual_explanation + when_this_helps`
  (1024-d unit-norm). **DONE: `data/enriched_core.parquet`** (gitignored; ~35s on GPU).
- `enrichment/writeback_kb.py` — backs up + writes the 4 fields + new vectors into KB
  `corpus.parquet` + `embeddings.f32`. **DONE (applied; `.pre_enrich.bak` backups exist).**
- `enrichment/watch_gold.py` — live batch/gold monitor (used during gold gen).

### What's DONE (A1→A4, all shipped)
1. Counseling core defined (17,808). 2. Claude (Opus 4.8) gold seed 1,490/1,490, offline only.
3. QLoRA-tuned Gemma 4 26B MoE on the gold. 4. Enriched 17,804/17,808 rows (99.98%), embedded,
written back into the KB, **pushed to private HF** (`push_hf` must be run by the human — the
harness blocks the bulk external upload for the agent).

### What's NEXT
1. **AUDIT the shipped enrichment** (deferred by owner — see below) before relying on it for V1.
2. **V1 backend** — scaffold the FastAPI pipeline (safety → understand+plan → BGE-M3 retrieve
   over the counseling core + Postgres exact lookup + rerank → Claude Sonnet 4.6 generate →
   deterministic citation verify). Load the enriched KB from HF.
3. **V2 later** — QLoRA + **DPO** the *generation* Gemma on preference pairs (much of it
   collected from V1 usage). Enrichment did NOT use DPO (gold-target SFT task; DPO is for persona).
4. Enrich the widen/shastrarth tiers + the 4 null rows if/when wanted (optional).

### AUDIT — data audit PASSED (2026-06); only retrieval-lift remains (folds into V1)
Audited completeness + integrity + quality, all clean:
- **Completeness** ✓ 231,940 rows · 17,808 core · 17,804 enriched · 4 known nulls · 0 non-core
  rows wrongly touched · all fields filled.
- **Integrity** ✓ all 17,804 new vectors unit-norm + changed vs backup; 5,000/5,000 sampled
  non-enriched rows byte-identical to backup (rest of KB uncorrupted); pushed HF file correct.
- **Quality** ✓ contextual_explanation 100% unique (no collapse), none truncated, zero JSON/
  prompt-echo artifacts; sampled biography tail + Bhashyam + Upanishads faithful, Gujarati natural.
- **4 null rows** (`vachanamrut_166` + 3 sw_lit chunks) — accepted, retrieval-only.
- **Retrieval lift** — NOT yet measured (needs a retriever): do at V1 time, before/after, using
  the old embeddings preserved in `embeddings.f32.pre_enrich.bak`.
- **Provenance / V2 purity** — enrichment is Gemma-generated but the adapter was QLoRA-tuned on
  **Claude-Opus gold**, so it's *Claude-bootstrapped*. Runtime stays Gemma-only (claim intact);
  if strict "never-Claude" V2 purity is later required, regenerate with a non-Claude-gold adapter
  (fully reversible — source text untouched, KB `.pre_enrich.bak` backups kept).
- **KB repo CLAUDE.md** (`../satsangai/CLAUDE.md`) still says "enrichment NOT done" — now stale;
  update it (that repo is on branch `kb-integrity-remediation`).

## V1 backend — run & deploy (`api/`)
Pipeline: `safety` → `understand` (Sonnet JSON) → `retrieve` (BGE-M3 + tradition filter +
`rerank` bge-reranker-v2-m3) → `generate` (Sonnet, streaming, `[P#]`-grounded) → `verify`
(deterministic) → memory (short-term always; long-term gated by `is_sensitive`). Chat UI at
`/` (`api/web/index.html`); SSE `/chat`; `/health`.

```bash
# build the retrieval index from the enriched KB (once)
python -m api.build_index
# run (in-memory stores; default)
source ~/.zshrc && HF_HUB_OFFLINE=1 uvicorn api.main:app --host 0.0.0.0 --port 8000
# tests + evals
python -m api.tests.test_safety_memory
source ~/.zshrc && HF_HUB_OFFLINE=1 python -m eval.run_eval        # + eval.topic_switch / retrieval_lift
```
Storage is swappable via `SATSANG_STORE` (`memory` default | `postgres`):
```bash
docker compose up -d                                                # pgvector Postgres (:5433)
SATSANG_DATABASE_URL=postgresql://postgres:satsang@localhost:5433/satsang python -m api.db.load_pg
SATSANG_STORE=postgres SATSANG_DATABASE_URL=... uvicorn api.main:app --port 8000
```
- `api/store.py` factories pick memory vs `api/pg.py` (pgvector) by config — verified at parity
  (10/10 search overlap; sensitive-data gate enforced in PG too). Schema: `api/db/schema.sql`.
- Toggles (env): `SATSANG_RERANK=0` off; `SATSANG_FAITHFULNESS_GUARD=1` (non-streaming, revises
  unfaithful claims); `SATSANG_HELPLINES_VERIFIED` (India-core verified, default on);
  `SATSANG_EMBED_DEVICE=cuda` for throughput. Models: `claude-sonnet-4-6` gen/plan, Opus judge in evals.

## Environment / gotchas
- GPU box (RTX PRO 6000 **Blackwell**, 96 GB). HF token cached in `$HF_HOME`.
- **Blackwell + Gemma 4 MoE:** torch 2.8's `grouped_mm` MoE kernel is Hopper-only (cc==9.0);
  must pass `experts_implementation="eager"` to `from_pretrained` for both inference AND
  training, or generation/forward crashes. Needs **transformers ≥5.x** (gemma4 arch) and a
  separately-downloaded `chat_template.jinja`. PEFT can't wrap the vision tower's
  `Gemma4ClippableLinear`; target only `language_model` `Linear4bit` projections.
- HF cache `.locks/` is root-owned → `mkdir -p $HF_HOME/hub/.locks` once (done), or
  `HF_HUB_OFFLINE=1` for already-cached models (BGE-M3, Gemma 4). hub 1.19 vs surya-ocr
  conflict in the KB repo is harmless (OCR already done).
- **`ANTHROPIC_API_KEY` lives in `~/.zshrc`**; non-interactive shells don't auto-source it
  and editing shell profiles is blocked here — **prefix key-needing commands with
  `source ~/.zshrc 2>/dev/null;`**. Used for offline gold + (later) V1 generation only.
