# SatsangAI — AI + Application Layer

This repo is the **AI + application layer** for SatsangAI: a warm "saint" companion
that helps people with real, messy life problems through the wisdom of Hindu +
Swaminarayan sacred texts — problem-first, zero-hallucination, multilingual.

GitHub: https://github.com/aarsh-adhvaryu/satsangAI

The **knowledge base is a separate, finished project** — do NOT rebuild it here.
This repo *consumes* it and adds enrichment, retrieval, generation, memory, and the API.

## ⏯️ CURRENT STATUS (2026-07) — resume here
**V1 AND V2 are both complete, shipped, and evaluated. Both counseling stacks work.**
Everything is committed to GitHub `main`. Full ongoing detail: project memory `satsangai-project-state`.
- **KB DONE + audited:** 231,940 rows, 17,804 counseling-core enriched (Gemma 4 26B MoE, QLoRA on
  Claude-Opus gold, runtime Claude-free), pushed to HF `aarsh-adhvaryu/satsangai-kb` (private).
  Re-audited 2026-07 CLEAN; 14 non-scripture OCR chunks excluded from core (`excluded_ids`).
- **V1 (Claude+RAG) DONE:** `api/` safety → understand → retrieve(+rerank) → generate → verify →
  memory; chat UI (polished: dark mode, citation chips, grounding status); Postgres-swappable.
- **V2 (from-scratch Gemma, Claude-FREE at runtime) DONE + SHIPPED:** SFT→on-policy-DPO on
  Gemma-4 26B (bf16 LoRA r64). Ship `v2/data/gemma4-v2-dpo2-lora`; won every blinded Opus
  pairwise judge. Serve via `SATSANG_GEN_BACKEND=gemma` (opt-in; Claude V1 is the default).
  KEY NEGATIVE RESULT: a 6-passage-context retrain LOST to the 1-passage dpo2 — do NOT retrain
  on multi-passage data. Synthetic-data quality has plateaued; only real user convos improve it.
- **The 6 proposal "partial" items COMPLETED (2026-07):** (1) formal 6-gate+RAGAS deploy eval
  `eval/six_gate.py`; (2) real **shastrarth** mode (SHASTRARTH_PERSONA + 8,173 Sanskrit-clean
  acharya-school rows in the index, tradition-filtered out of counseling); (3) dual-layer emotion
  (surface+underlying) in `understand.py`; (4) deployment layer — WebSocket `/ws`, observability
  `/metrics`+`/traces`, vLLM `v2/serve_vllm.py`; (5) Gujarati training-pair pipeline
  `v2/multilingual_pairs.py` (data ready; bilingual SFT is a GPU step); (6) safe-by-default
  regional/diaspora helpline mechanism `config/helplines.yaml` (numbers need human verify).
- **✅✅ COUNSELING DEPLOY CONFIRMED AT k=3 (`eval/six_gate_k3.json`, 65 probes x 3 draws = 195):**
  hallucination **1.000** · persona **1.000** · sycophancy **1.000** · emotional **1.000** ·
  scripture **1.000** · RAGAS 0.920 over n=130 counseling draws → **DEPLOY ✓**. This supersedes the
  single-sample run and is the number to quote. STABILITY (the reason k=3 exists): 19 probes flipped
  verdict between identical draws — **6 hallucination + 11 scripture + 2 sycophancy, ALL of them
  shastrarth**. The 41 pure-counseling probes flipped on ZERO gates (RAGAS spread mean 0.043). So the
  ±0.13 run-to-run noise measured earlier is a SHASTRARTH phenomenon (unreadable grounding → unstable
  behaviour); counseling was reproducible all along. Also learned: `understand()` is an LLM call, so
  **mode routing itself is stochastic — 5/65 probes changed mode between draws.**
  Shastrarth at k=3 stays REJECT (hallucination 0.846, scripture 0.769).
- Prior single-sample run (superseded, kept for history): `eval/six_gate_v1b_rescored.json`.
  hallucination 1.0 · persona 1.0 · sycophancy 0.976 · emotional 1.0 · scripture 1.0 · RAGAS 0.908.
  The earlier 0.929 `emotional` REJECT was a DETECTOR BUG, not a defect: all 3 counseling misses were
  regex false positives — `take (an?|\d)` matched "just take a breath" (x2) and bare `dosage` matched
  inside an explicit *refusal* ("I'm not able to suggest any medication or dosage"). Fixed in
  `eval/six_gate.py` (`_MED_INSTRUCT` requires a real substance after "take"; `_MED_REFUSAL` cancels
  matches in refusal sentences; "prescribe" only counts near a medication noun — it is also ordinary
  theological English, "the mantra prescribes an orientation of devotion"). Verified on 12 true/false
  cases; 0/42 counseling and 0/23 shastrarth replies now flagged. **Lesson (3rd time): regex detectors
  are the #1 source of false eval signal here — always read the flagged text before believing a gate.**
  Re-derive any saved run offline (no API/GPU) with `python -m eval.rescore --in <run>.json`.
  RESIDUAL (not a gate failure, worth a look): the newborn reply gives real infant-care technique
  (the "5 S's", positioning) that no detector catches, and is independently weakest — sycophancy=False,
  faithfulness 0.70, practical advice ungrounded in any passage. Scope question, not hallucination.
- **SHASTRARTH still REJECT** (hallucination 0.78, scripture 0.83). ROOT CAUSE MEASURED — and it is
  **NOT "untranslated Sanskrit" (that earlier claim was wrong on two counts)**:
  * All 8,173 acharya-school rows have `original` only — 0% translation/transliteration/word_meanings/
    contextual_explanation, `commentaries` an empty 2-char placeholder. That part holds.
  * BUT only 5,301 (65%) are Devanagari. **2,558 (31%) are already ENGLISH** — OCR'd English editions:
    shankara_brahmasutra_bhashya (924), ramanuja_gita_bhashya (511), madhva_chandogya_bhashya (1,182),
    vedartha_sangraha (67), ramanuja_gadya_traya (75) are all dev-ratio ~0.00. Only ramanuja_sribhashya
    (2,883, dev 0.99) and vallabha_anubhashya (2,531, dev 0.96) are genuine Sanskrit. For the English
    third the need is **OCR CLEANUP, not translation** (`BRAHMA-SOTRA-BHASYA`, `U~d`→Upaniṣad, running
    headers like `[Lill.lt`). Splitting the job by source is mandatory — one pipeline won't fit both.
  * **THE REAL DIFFERENTIATOR IS ENRICHED vs UNENRICHED, not language.** 15,318/17,794 non-school rows
    ALSO have empty `translation` (incl. 12,491 swaminarayan) yet counseling scores 6/6 — because
    15,315 of them carry `contextual_explanation`. `_passages_block` emits `text:` (falls back to
    `original`) AND `meaning:` (enrichment); counseling passages arrive with a populated `meaning:`,
    school passages have neither field. So shastrarth's fix is the ENRICHMENT layer (A1-A4 pipeline).
  * ⚠️ `eval/shastrarth_translate.py` therefore measures a PROXY: it fills `text:` but leaves `meaning:`
    empty, i.e. "readable source, still unenriched" = a LOWER BOUND. A flat result does NOT rule out
    enrichment working.
  * **RESULT (23 probes, `eval/shastrarth_translate.json`): NO gate flipped** — hallucination
    0.913→0.957, scripture 0.739→0.739, ragas 0.831→0.817, sycophancy 1.000→0.913. 5 probes improved
    (mean +0.209), 6 worsened (mean −0.215), 12 flat: symmetric around zero = noise, not effect.
- **🚨 EVAL METHODOLOGY DEFECT (found 2026-07-25) — ALL gate numbers are single high-temperature
  samples.** `api/generate.py` sets no `temperature`, so evals draw at the API default 1.0. Measured
  run-to-run noise on the SAME 23 shastrarth probes with NOTHING changed: hallucination ±0.130,
  scripture ∓0.087, sycophancy ±0.043, ragas ∓0.044 — i.e. **the noise floor exceeds every treatment
  effect we have tried to measure.** Consequences: (a) the shastrarth translation experiment is
  underpowered and cannot answer its question; (b) **the COUNSELING DEPLOY 6/6 is not yet
  reproducible-grade** — its `hallucination` threshold is 1.00, which one unlucky sample in 42 breaks.
  FIX BEFORE ANY FURTHER GATE WORK: k-sample (k=3) per probe and average for deploy gates (faithful to
  the served temperature-1.0 distribution), and `temperature=0` for A/B experiments (removes variance
  when comparing two configs). Do not spend GPU on enrichment until the eval can resolve the effect.
  ⚠️ Do NOT "fix" this with a runtime translate/reasoning node: that puts an LLM UPSTREAM of the
  deterministic citation verifier, so a faithful reply to a mistranslation would pass unchecked.
  DECIDE FIRST with `python -m eval.shastrarth_translate --limit 6` (CPU+API, no GPU) — it A/Bs raw
  vs translated grounding on the same passages and prints which gates flip.
  DEEPER GAP: §18 wants Swaminarayan *alongside* Shankara/Ramanuja/Madhva on the same sutra, but the
  7 school sources have no per-verse cross-school alignment. Translation won't create that.
- **✅ BILINGUAL V2 CHAIN COMPLETE (2026-07-25 13:31).** `gemma4-v2-bi-sft-lora` (579 steps, eval
  0.94) → on-policy negatives (6,439 pairs) → `gemma4-v2-bi-dpo-lora` (383 steps, loss 0.688→0.073,
  eval 0.244→0.073, accuracies 0.362→0.99 — healthy curve matching shipped dpo2, NOT the degenerate
  dpo1 shape; rewards/chosen −2.84 = the known likelihood displacement, covered by `api/verify`).
  Verified NO language shortcut: chosen/rejected share a language in 98% of Gujarati pairs, 100% of
  English, lengths balanced, 81% of negatives on-policy.
  **VERDICT SO FAR — near parity, NOT yet promotable:**
  * English 6-gate: **dpo2 12/12, bi 12/12 TIED** (bi first scored 11/12; that was a 4th REGEX
    FALSE SIGNAL — `_PUSHBACK` matched "gently" but not "one **gentle** question… with care, not
    judgment: is there any part of you…", while dpo2 passed on the filler word "honestly". Detector
    fixed in `v2/eval_gates.py`; re-scoring the saved replies gives 12/12 for both.)
  * Blinded Opus pairwise (12 probes): overall tie 5 / dpo2 4 / bi 3; **loving_pushback dpo2 4-0-8
    (dpo2's signature strength, bi never wins it)**; faithfulness bi 4-2-6; warmth/depth/appropriateness
    ~tied. So bi trades a little pushback for a little faithfulness.
  * **GUJARATI MEASURED → BILINGUAL IS A CLEAN NEGATIVE RESULT. KEEP `dpo2`. DO NOT PROMOTE `bi`.**
    (`v2/data/gate_results_bi_gu.json`, `pairwise_bi_gujarati.json`, 11 Gujarati probes.)
    - **The run's premise was false: `dpo2` was ALREADY fluent in Gujarati** — mean gu_script 0.98,
      **0/11 answered in English**. There was no language gap to close (Gemma 4 is natively
      multilingual and the KB's Gujarati enrichment already reaches the model via retrieval).
      `bi` 0.99 / 0-11. Deterministic gates identical: **7/11 both**.
    - Blinded Opus pairwise ON GUJARATI: overall **dpo2 5 / bi 2 / tie 4**; **warmth dpo2 9 / bi 0**;
      depth dpo2 6 / bi 4; faithfulness 3-3-5; loving_pushback 3-2-6. dpo2 wins its own home turf
      AND the language it was never trained for.
    - **ROOT CAUSE of bi's warmth collapse (a data-design flaw in `v2/multilingual_pairs.py`):** its
      `_REPLY_SYS` says *"Be brief and human; never preach"* with `max_tokens=600`, so Gujarati
      `chosen` averaged **548 chars vs 1,299 for English** — 42% of the depth. The bilingual SFT
      learned terseness, and terseness reads as coldness: bi replies average 602 chars vs dpo2's 764.
      IF ANYONE RETRIES BILINGUAL: generate Gujarati `chosen` at the SAME depth as English first.
    - **This is the THIRD data-engineering attempt to beat dpo2 that has failed** (6-passage retrain,
      naturalized problems, now bilingual). The lesson holds: synthetic data has plateaued; only real
      user conversations will move it.
- (historical) BILINGUAL V2 run started (~3.5h `&&` chain, resumable):
  `multilingual_pairs` (Gujarati) → cat with 1-passage `pairs.jsonl` → `pairs_bilingual.jsonl` →
  `sft_train`(→`gemma4-v2-bi-sft-lora`) → `onpolicy_negatives` → `dpo_train`(→`gemma4-v2-bi-dpo-lora`)
  → `eval_gates` + `judge_pairwise` (dpo2 vs **bi**). **NEW SESSION: read `v2/data/gate_results_bi.json`
  + the pairwise output — if `bi` beats/ties dpo2 without English regression, promote it
  (`SATSANG_GEMMA_ADAPTER=v2/data/gemma4-v2-bi-dpo-lora`); else keep dpo2. If the chain died, re-run
  the same command (every stage is resumable).** Full command in project memory `satsangai-project-state`.
- **NEXT after bilingual verdict:** (a) OPTIONAL vLLM serve (`v2/serve_vllm.py`); (b) human-verify
  helpline numbers in `config/helplines.yaml`; (c) THE REAL MILESTONE = **deploy to collect real
  conversations** (only path past the synthetic-data plateau + feeds continuous DPO).
- **PROPOSAL GAP AUDIT (2026-07) — what the spec promises that does NOT exist:**
  1. **Response modes: 2 of 6 built.** `understand.py` enum is only `["counseling","shastrarth"]`.
     Missing: §5.2 verse explanation (KB already has every field), §5.3 creative writing —
     poems/prayers/kirtans + §19 attribution rules, §5.4 satsang speeches, §5.5 daily wisdom.
     NOTE §9 names the poem-for-a-late-mother as a *defining* success criterion — creative gen is
     not decoration. 5.2 + 5.3 are the cheapest high-value builds left.
  2. **V2 is not Claude-free end-to-end.** §10 specifies a second model, **Gemma 4 E4B**, for the
     understand/plan/emotion node; today `SATSANG_GEN_BACKEND=gemma` swaps ONLY generation and
     `api/understand.py` still calls Sonnet, so V2 can't run without an Anthropic key. Most
     load-bearing item for the §23 from-scratch claim; E4B needs no fine-tuning.
  3. **§7 "control is absolute" unimplemented** — no view/edit/delete per memory item, no pause, no
     export/clear; `api/main.py` has zero memory endpoints. Also missing *interaction memory*
     (length/language/story-vs-direct preferences). User-rights guarantee ⇒ launch blocker.
  4. **§4.2/§12.3 audio ingestion** (pravachan ASR + speaker ID + verse matching + human review) and
     **§4.3/§17 morphology** (dhatu/case analysis, analogy engine): untouched, genuine projects.
  5. **§20.3 wants 200+ probes**, sign-off used 65. **§21 CI/CD eval gate** (GitHub Actions blocking
     deploy on RAGAS drop) not built — that's what makes §8 automatic instead of manual.
  6. **§8/§29 flywheel** (consent → capture → preference pairs → continuous DPO) not wired; gated on deploy.
  Deliberate deviations, no action: BGE-M3 over Voyage (required for vector alignment), numpy+pgvector
  over ChromaDB, vanilla HTML over React/Next (revisit if the §28 memory panel is built).
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
