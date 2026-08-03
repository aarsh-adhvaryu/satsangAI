# SatsangAI

A companion that answers real, messy life problems through Hindu and Swaminarayan
scripture — **problem-first, multilingual, and grounded: every scriptural reference is
verified against the corpus by code, never by another model.**

This repo is the model and application layer. The corpus is a separate project:
**[satsangAI_KB](https://github.com/aarsh-adhvaryu/satsangAI_KB)** — 228,121 chunks of
Sanskrit, Gujarati, Hindi and English scripture.

---

## What it is

A **fine-tuned Gemma 4 26B MoE** (QLoRA SFT → DPO) serving a retrieval-grounded pipeline
with deterministic safety and citation verification. It runs with **no commercial API at
runtime** — planning, generation and verification are all local.

```
safety classifier          deterministic, runs FIRST, cannot be bypassed
  ↓
understand + plan          emotion (surface + underlying), mode, search queries
  ↓
retrieve                   BGE-M3 → tradition-filtered search → cross-encoder rerank
  ↓
generate                   tuned Gemma, streaming, grounded only in retrieved passages
  ↓
verify                     every [P#] resolved against the corpus — deterministic
  ↓
memory                     session history always; long-term facts behind a hard
                           exclusion gate for self-harm / abuse / medical disclosures
```

Six response modes: **counseling · teaching · verse explanation · creative writing ·
out-of-domain refusal · shastrarth** (parked — the corpus no longer holds the acharya
schools it needs).

---

## Results

Deploy gate — 99 adversarial probes × 3 independent draws (297 responses), deterministic
gates, no API:

| mode | n | hallucination | emotional |
|---|---|---|---|
| counseling | 123 | 1.000 | 1.000 |
| teaching | 63 | 1.000 | 1.000 |
| verse | 63 | 1.000 | 1.000 |
| creative | 36 | 1.000 | 1.000 |
| out_of_domain | 12 | 1.000 | 1.000 |

**0 fabricated citations across 297 responses. 0/99 probes changed verdict between draws**
— the result is reproducible rather than a favourable sample. Mode routing: 99/102.

*Scope:* these are the correctness gates — citations, mode contracts, attribution,
medical-instruction, routing. The judged quality gates (persona, sycophancy, RAGAS) were
measured on the earlier Claude-backed pipeline, not on the shipped local model.

---

## Notable findings

Three results that shaped the system, all reproducible from this repo:

**Quantization silently does nothing on this architecture.** A "4-bit" checkpoint came out
46 GB instead of ~13 GB. Reading tensor dtypes from the safetensors header showed 47.2 GB
still bf16 against 1.14 GB quantized — **2.4% of weights**. `transformers` stores Gemma-4
experts as fused 3-D parameter tensors, and bitsandbytes only replaces `nn.Linear`. The
config was obeyed; it had almost nothing to act on.

**The enrichment layer gave no measurable retrieval lift.** A blind A/B against the previous
embeddings came out 5-5. BGE-M3 is strongly cross-lingual, so translation-level embeddings
already matched English queries. Enrichment's real value is generation grounding, not
retrieval — a conclusion that only survived because it was measured rather than assumed.

**Single-sample evaluation was unusable.** Re-running identical probes moved hallucination
±0.13 — larger than every treatment effect the project had measured. All deploy decisions
now use k=3 sampling with per-mode segmentation.

---

## Run it

```bash
# local model, no API key required anywhere (GPU)
bash serve.sh

# the deploy gate, zero API cost
python -m eval.six_gate --backend gemma --judge none --k 3 --out eval/run.json
python -m eval.watch_gates --out eval/run.json --total 297 --fails

# tests
python -m api.tests.test_safety_memory
python -m api.tests.test_verify_chapter_verse
```

Docker: `docker compose up --build` (CPU image, Claude backend). Postgres/pgvector behind
the `postgres` profile.

---

## Stack

Gemma 4 26B MoE · PEFT/QLoRA · TRL (SFT + DPO) · BGE-M3 · bge-reranker-v2-m3 ·
FastAPI (SSE + WebSocket) · PostgreSQL/pgvector · Docker · Transformers

## Layout

| path | what |
|---|---|
| `api/` | the serving pipeline — safety, routing, retrieval, generation, verification, memory |
| `v2/` | the from-scratch model: pair generation, SFT, DPO, quantization, evaluation |
| `enrichment/` | KB enrichment with a locally-tuned Gemma, and write-back |
| `eval/` | the 6-gate deploy pipeline, probes, judges, rescoring |
| `config/` | counseling-core manifest, helplines, emergency numbers |

Engineering notes, measured numbers and the failure log live in [CLAUDE.md](CLAUDE.md).

---

## Safety

Crisis detection is deterministic pattern matching that runs **before** any model call and
cannot be bypassed, including method-seeking phrasing ("how many … to not wake up") that an
intent-only classifier misses. Responses carry human-verified helplines (India verified;
regional and diaspora entries ship inert until confirmed). Disclosures involving self-harm,
abuse, trauma or medical conditions are never written to long-term memory — that gate applies
to user edits too, not just to extraction.

This is a companion for loneliness and life's difficulties. It is not a therapist, and it is
built to say so.
