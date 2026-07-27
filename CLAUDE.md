# SatsangAI — AI + Application Layer

A warm "saint" companion that helps people with real, messy life problems through Hindu +
Swaminarayan scripture — problem-first, zero-hallucination, multilingual.

GitHub: https://github.com/aarsh-adhvaryu/satsangAI
Full spec: `../.data/data/data/SatsangAI_Final_Complete_Proposal.md` (Part I vision,
Part II V2 from-scratch Gemma, Part III V1 Claude+RAG).

**The knowledge base is a separate, finished project — do NOT rebuild it here.**
This repo consumes it and adds enrichment, retrieval, generation, memory, and the API.

---

## ⏯️ STATUS (2026-07-27) — resume here

**The system runs end-to-end with NO Anthropic API.** Verified on an H100: 99 probes, no
`ANTHROPIC_API_KEY` in the environment, routing 34/34, zero contract breaches.

Latest gate — `eval/six_gate_final.json`, deterministic gates only (`--judge none`):

| mode | n | hallucination | emotional | verdict |
|---|---|---|---|---|
| counseling | 41 | **1.000** | 1.000 | **PASS** |
| verse | 12 | **1.000** | 1.000 | **PASS** |
| creative | 12 | **1.000** | 1.000 | **PASS** (7 written + 5 guard refusals) |
| out_of_domain | 4 | **1.000** | 1.000 | **PASS** |
| teaching | 30 | 0.967 | 1.000 | 1 miss (verbatim-shloka bait) |

Judged gates (persona / sycophancy / scripture / RAGAS) are **unscored** on this stack —
they need an API judge. The last full judged run gave counseling 6/6 at k=3
(`eval/six_gate_k3.json`): five gates at exactly 1.000, RAGAS 0.920.

**Work is on branch `feat/modes-and-claude-free` (pushed, 2 commits ahead of main).**

---

## 🚨 KNOWN ISSUES — read before deploying

1. **Silent failure on the crisis path.** `api/safety.py` wraps both YAML loads in
   `except Exception: return ""`. A malformed `helplines.yaml` or `emergency_numbers.yaml`
   silently drops every regional/country helpline with no warning anywhere. This is the one
   path where being wrong is unrecoverable — it should log loudly, not fail quietly.
2. **Two overlapping helpline configs.** `helplines.yaml` (diaspora block, `verified:false`)
   duplicates numbers now live in `emergency_numbers.yaml` (US 988, UK Samaritans, CA 988).
   Dedup happens only *within* `_country_appendix`, so flipping the diaspora block to
   `verified:true` would print 988 twice. Consolidate onto one file.
3. **Memory controls are unreachable.** 10 of 14 endpoints have no UI at all — `/memory`,
   `/memory/{id}` (PATCH/DELETE), `/memory/prefs`, `/memory/export`, `/feedback`. §7 promises
   "control is absolute" and §29 promises consent; both are currently unfulfilled promises
   while the app stores real personal disclosures. **Biggest genuine blocker.**
4. **Postgres breaks memory controls.** `PgMemoryStore` lacks `items/update/delete/clear`,
   so those endpoints return 501 under `SATSANG_STORE=postgres`. Deploy with the memory
   store, or add parity (~1h). Matters for any host with ephemeral disk.
5. **Config surface is split.** `SATSANG_COUNTRY` and `SATSANG_REGION` are read inline in
   `safety.py`, not declared in `config.py` like every other knob.
6. **Creative is safe but not good.** 1.000 is reached partly by refusing: ~42% of requests
   return an apology instead of a poem. Prompting is exhausted (three attempts, all inside
   noise) — lowering the refusal rate needs training data.
7. **Verse trades helpfulness for verifiability.** For verses with no stored `word_meanings`
   the guard strips the breakdown, even though Gemma's glosses were *correct*. Real fix is
   backfilling `word_meanings` in the KB.

---

## Architecture

```
safety (deterministic, FIRST, cannot be bypassed)
  -> understand + plan          api/understand.py   (Claude OR Gemma via api/llm.py)
  -> domain gate                in_domain=false -> out_of_domain, no retrieval
  -> verse lookup               api/verse.py        (deterministic citation match)
  -> creative detection         api/creative.py     (form + output language)
  -> retrieve                   api/retrieve.py     (BGE-M3 + tradition filter + rerank)
  -> generate                   api/generate.py     (persona per mode, [P#] grounded)
  -> verify                     api/verify.py       (deterministic; §19 for creative)
  -> memory                     api/memory.py       (short-term always; long-term gated)
```

**Two shared entry points — do not bypass them.** `pipeline.prepare()` owns routing and
retrieval; `pipeline.generate_reply()` owns every generation-time decision (creative guard,
verse guard, faithfulness guard, streaming). The eval calls both.

> This exists because the eval twice kept its own copy of a pipeline stage and so measured
> code the product does not run — first routing (scored out_of_domain 0/12 and verse 0/15),
> then generation (scored creative 0.417 with the §19 guard never executing). **Any new
> pipeline step goes in one of those two functions.** The durable fix is `--backend http`
> so the eval hits the running server and no second code path can exist.

### Modes

| mode | routed by | retrieval | notes |
|---|---|---|---|
| `counseling` | planner (default) | enriched core, tradition-filtered | the shipped product |
| `teaching` | planner | same as counseling | learners; NOT shastrarth |
| `verse` | **deterministic** `verse.parse_reference` | verse pinned as `[P1]` | ~1,187 addressable verses |
| `creative` | **deterministic** `creative.detect_form` | enriched core | §19 guard + refusal |
| `out_of_domain` | planner `in_domain=false` | **none** | honest refusal, no scripture |
| `shastrarth` | **user selection only** | all schools, unenriched | OFF by default, fails 2 gates |

---

## Core principles (from the proposal)

- **Problem-first, not scripture-first.**
- **Zero hallucination** — every `[P#]` verified by deterministic DB lookup, never by an LLM.
- **Saint persona** — warm, never lectures, pushes back lovingly (no sycophancy).
- **Safety first** — deterministic crisis classifier runs BEFORE any LLM.
- **Tradition-aware** — never mix schools in counseling; full breadth only in shastrarth.
- **Memory with hard sensitive-data exclusion** — self-harm/abuse/trauma/medical/criminal
  disclosures are NEVER written to long-term memory, including via user edits.

---

## Run it

```bash
# Claude backend (needs ANTHROPIC_API_KEY in ~/.zshrc; credits currently exhausted)
source ~/.zshrc && HF_HUB_OFFLINE=1 uvicorn api.main:app --port 8000

# Claude-FREE: everything on the local GPU, no API key at all (~4.5 min model load)
env -u ANTHROPIC_API_KEY SATSANG_UTILITY_BACKEND=gemma SATSANG_GEN_BACKEND=gemma \
  SATSANG_EMBED_DEVICE=cuda HF_HUB_OFFLINE=1 uvicorn api.main:app --port 8000

# deterministic gate, zero API cost (~65 min for 99 probes)
env -u ANTHROPIC_API_KEY SATSANG_UTILITY_BACKEND=gemma SATSANG_GEN_BACKEND=gemma \
  SATSANG_EMBED_DEVICE=cuda HF_HUB_OFFLINE=1 \
  python -m eval.six_gate --backend claude --judge none --k 1 --out eval/run.json

python -m eval.rescore --in eval/run.json   # re-derive verdicts free after a detector fix
python -m api.tests.test_safety_memory      # safety + memory + crisis regression
python -m api.build_index                   # rebuild the retrieval index from the KB
```

### Environment

| var | default | meaning |
|---|---|---|
| `SATSANG_GEN_BACKEND` | `claude` | `gemma` = saint reply from the tuned adapter |
| `SATSANG_UTILITY_BACKEND` | `claude` | `gemma` = plan + memory extraction local. **Both = no API** |
| `SATSANG_UTILITY_MODEL` | `""` | empty reuses the generation model with the adapter disabled |
| `SATSANG_GEMMA_ADAPTER` | `dpo2` | which V2 adapter serves |
| `SATSANG_EMBED_DEVICE` | `cpu` | `cuda` — retrieval is the CPU bottleneck |
| `SATSANG_SHASTRARTH` | `0` | `1` offers shastrarth as a user-selectable mode |
| `SATSANG_STORE` | `memory` | `postgres` (breaks memory controls — see issue 4) |
| `SATSANG_COUNTRY` | unset | ISO-2 code for country helplines (read in `safety.py`) |
| `SATSANG_REGION` | unset | regional helpline block (read in `safety.py`) |
| `SATSANG_RERANK` / `SATSANG_FAITHFULNESS_GUARD` | `1` / `0` | toggles |
| `SATSANG_GEN_MODEL` / `SATSANG_PLAN_MODEL` | `claude-sonnet-4-6` | Claude model ids |
| `SATSANG_EMBED_MODEL` / `SATSANG_RERANK_MODEL` | BGE-M3 / reranker | model ids |

**GPU rule: ONE 52 GB model at a time on an 80 GB card.** `api/llm._require_free_vram()`
fails in <1s with the fix rather than OOM-ing after a long load.

---

## Deploy

`Dockerfile` builds a self-contained image (8.7 GB; build context 188 MB). It copies
`api/ config/ enrichment/ v2/` — **`v2/` is required on the Gemma path** (`api/generate`
imports `v2.train_config`); omitting it crashed the container while working outside it.
`v2/data/` (18 GB of adapters) is excluded and must be mounted:

```bash
docker run --gpus all -p 8000:8000 \
  -e SATSANG_UTILITY_BACKEND=gemma -e SATSANG_GEN_BACKEND=gemma -e SATSANG_EMBED_DEVICE=cuda \
  -v $PWD/v2/data/gemma4-v2-dpo2-lora:/app/v2/data/gemma4-v2-dpo2-lora:ro \
  -v $HOME/.cache/huggingface:/models:ro  satsangai:modes
```

**Hosting:** the UI is a 17 KB file served by FastAPI at `/` — same-origin, no build step,
no CORS. Do not split it onto Vercel. Options: Lightning Studio (GPU already paid for,
£0 extra, best for collecting first conversations) · HF Spaces Docker SDK + external
Postgres (ephemeral disk otherwise loses everything) · Railway/Fly + Claude (per §25).
Always-on GPU for the 52 GB model is ~$1–2k/month; 4-bit (~26 GB) could change that but
is **untested for serving**.

---

## Knowledge base (consume, don't recreate)

- HF dataset (PRIVATE): `aarsh-adhvaryu/satsangai-kb` — **231,940 records**, BGE-M3
  embeddings on every row. KB repo: `../satsangai`.
- **Counseling core = 17,794 rows**, 17,804 enriched (Gemma 4 26B MoE, QLoRA on Claude-Opus
  gold, runtime Claude-free). Shastrarth adds 8,173 acharya-school rows (unenriched).
- **Script reality** (measured, drives `api/verse.py`): `original` is Gujarati 8,466 /
  **Latin 7,208** / Devanagari 1,476 / Kannada 643. Shikshapatri and Satsang Diksha ship
  already-romanised. A hardcoded Devanagari transliteration silently no-ops.
- **Coverage gaps**: transliteration 5.6%, `word_meanings` 3.9%, translation 13.9% —
  `contextual_explanation` is 100% and is what makes counseling work.
- **Verse-addressable only**: Gita (719, full word-by-word), Yoga Sutras (195), Vachanamrut
  (273). Everything else is page-chunked and cannot resolve by verse number.

### Corrections to the proposal (apply these)
- Query embeddings MUST use BGE-M3, not Voyage/OpenAI — otherwise vectors don't align.
- Default counseling retrieval = the curated core, not the full 231k (narrative floods it).
- Enrichment engine = **local Gemma only**; Claude gold is offline-bootstrap only.

---

## V2 (the from-scratch model)

**`v2/data/gemma4-v2-dpo2-lora` is final and shipped.** It won every blinded pairwise judge.

Negative results worth not repeating:
- **6-passage retrain LOST** to 1-passage dpo2 — more context diluted the reply.
- **Bilingual retrain LOST** — dpo2 was *already* fluent Gujarati (gu_script 0.98, 0/11
  answered in English). Root cause of the loss: `multilingual_pairs.py` said "be brief",
  so Gujarati `chosen` averaged 548 chars vs English 1,299 — the model learned terseness,
  and terseness reads as coldness (warmth 9-0 against it).
- **Three data-engineering attempts have failed to beat dpo2.** Synthetic data has
  plateaued. Only real user conversations will move it.

---

## What's left, by priority

**Before deploying**
1. Frontend: memory panel, consent, feedback buttons (issue 3) — CPU only, no API.
2. Fix the crisis-path silent failure (issue 1) and consolidate helpline configs (issue 2).
3. Postgres parity for memory controls (issue 4) — or deploy with the memory store.
4. Helpline spot-check for the countries you expect users in.
5. Backend decision: Claude (per-conversation cost) vs Gemma (always-on GPU).

**After deploying — the actual milestone**
6. Collect real conversations with consent + feedback. §29's flywheel; the only path past
   the synthetic-data plateau.

**Proposal features not built**
- §5.5 daily wisdom (deferred — needs a scheduler and delivery channel)
- §4.2/§12.3 audio ingestion (pravachan ASR, speaker ID, verse matching) — untouched
- §4.3/§17 morphology (dhatu/case analysis, analogy engine) — untouched
- §10 Gemma 4 E4B utility model — currently the 26B base with the adapter disabled, which
  works; the E4B id was never confirmed or cached
- §20.3 wants 200+ probes; we have 99
- §21 CI/CD eval gate (GitHub Actions blocking on RAGAS drop) — no workflows exist
- §21 vLLM serving — `v2/serve_vllm.py` exists, never run
- §18 shastrarth — parked; fails hallucination 0.78 / scripture 0.83 because its 8,173
  school rows are unenriched. Root cause measured; fix is enrichment, not prompting.

**KB work (owner: last)** — `word_meanings` + transliteration backfill, shastrarth
enrichment, per-verse cross-school alignment for real §18 comparison.

---

## Hard-won lessons

- **Regex detectors are the #1 source of false eval signal here.** Four separate incidents
  in one day: `take (an?|\d)` matched "just take a breath"; `dosage` matched inside a
  refusal; `_PUSHBACK` missed "gentle" while passing on filler "honestly"; the verse gate
  fired on its own honest disclaimer. **Always read the flagged text before believing a
  gate.** Prefer sharing the product's own detector over writing a second one.
- **Single-sample gate numbers are unreliable.** Measured run-to-run noise on identical
  shastrarth inputs: hallucination ±0.13. Use `--k 3` for any real decision.
- **Segment by mode.** A combined average once read REJECT while counseling passed all six.
- **Live probes beat unit tests** for this system — the crisis method-seeking gap
  ("how many paracetamol…") and both pipeline-drift bugs were found by running real turns.
- **Prompting has limits.** Three creative-persona revisions all landed inside noise; a
  deterministic guard fixed it in one attempt.

## Environment gotchas

- GPU: H100 80 GB (Hopper `grouped_mm`) — Blackwell needs `experts_implementation="eager"`.
- `ANTHROPIC_API_KEY` lives in `~/.zshrc`; non-interactive shells don't source it, so prefix
  with `source ~/.zshrc 2>/dev/null;`. **Credits are currently exhausted.**
- Lightning terminals run inside `screen` — jobs survive a disconnected laptop.
- `pytest` is not installed; run test modules directly.
