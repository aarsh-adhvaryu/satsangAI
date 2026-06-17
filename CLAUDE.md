# SatsangAI — AI + Application Layer

This repo is the **AI + application layer** for SatsangAI: a warm "saint" companion
that helps people with real, messy life problems through the wisdom of Hindu +
Swaminarayan sacred texts — problem-first, zero-hallucination, multilingual.

GitHub: https://github.com/aarsh-adhvaryu/satsangAI

The **knowledge base is a separate, finished project** — do NOT rebuild it here.
This repo *consumes* it and adds enrichment, retrieval, generation, memory, and the API.

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

## Enrichment pipeline — STATUS (2026-06, in progress)
Build order: **V1 (Claude + RAG) first; V2 (Gemma) parallel.** Currently building the
enrichment layer (the proposal's primary retrieval target). Decisions locked with the
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
  `Linear4bit` targets, completion-only loss. **Validated** (smoke passed). NOT yet run for real.
- `enrichment/watch_gold.py` — live batch/gold monitor.

**NEXT (all need the GPU ON; paused here by owner to avoid idle GPU credits):**
1. `python -m enrichment.qlora_train --epochs 3` (~2–3 h).
2. Benchmark batched enrich throughput, then bulk-enrich the **full 17,808 core** via
   **transformers** (not vLLM — reliable on Blackwell; no install gamble). Wire the result
   into KB `../satsangai/pipeline/enrich.py`'s `local` backend.
3. Re-embed enriched rows on `contextual_explanation + when_this_helps` (BGE-M3), write
   back to the KB, hot-swap the retrieval index, push to private HF.
4. V1 backend (FastAPI pipeline above) in parallel; upgrade to enriched embeddings when ready.

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
