# SatsangAI — AI + Application Layer

An AI companion that answers real life problems through Hindu and Swaminarayan scripture:
problem-first, warm, and **grounded — every scriptural reference is checked against the
corpus by code, not by another model.**

This repo is the model + application layer. The corpus lives in
[satsangAI_KB](https://github.com/aarsh-adhvaryu/satsangAI_KB) — consume it, don't recreate it.

Rewritten 2026-08-03 against a full audit. Every number here was measured, not carried
forward; where a claim is untested it says so.

---

## ⏯️ STATUS — resume here

**V2 (the from-scratch Gemma) is the product and it passes its deploy gate.**

`eval/six_gate_gemma_k3.json` — 99 probes × 3 draws = 297, `--backend gemma --judge none`:

| mode | n | hallucination | emotional |
|---|---|---|---|
| counseling | 123 | 1.000 | 1.000 |
| teaching | 63 | 1.000 | 1.000 |
| verse | 63 | 1.000 | 1.000 |
| creative | 36 | 1.000 | 1.000 |
| out_of_domain | 12 | 1.000 | 1.000 |

`deploy: true`, `failing_modes: []`, routing 99/102, and **0/99 probes flipped verdict
between draws** — the result is reproducible, not a lucky sample.

**What that does and does not prove.** `--judge none` scores only the deterministic gates:
citations, mode contracts, §19 attribution, medical-instruction, routing. Persona,
sycophancy, scripture_accuracy and RAGAS are **unscored** — they need the Opus judge and
the owner's API credits are exhausted. So: correctness is proven at k=3; *quality* is not
measured on the shipped model.

**Serving:** bf16 on an H100 (~49 GB). `bash serve.sh`. Quantization is a dead end here —
see the MoE finding below.

---

## 🚨 KNOWN ISSUES — read before deploying

1. **Quantization does not work on this architecture.** bitsandbytes quantized 2.4% of the
   weights. Do not retry it with bnb. Details below.
2. **Quality gates are unmeasured on V2.** Everything judged (persona/sycophancy/RAGAS) was
   scored on the Claude-backed V1, not on the Gemma model that ships.
3. **§17 morphology has no KB storage**, so a grammar breakdown is always recalled, never
   retrieved. Guarded (`verse.claims_grammar`), not solved.
4. **Only 10.2% of the corpus is served** — 23,254 of 228,121 rows. The rest (mahabharata
   73k, itihasa 92k, valmiki 21k) is unenriched and invisible to retrieval.
5. **Shastrarth has zero data.** All four acharya traditions were removed from the KB as
   uncitable page-scans. The mode cannot be enabled from this corpus at any config setting.
6. **Helpline numbers** are human-verified for India only; regional/diaspora entries ship
   `verified: false` and are inert until a human confirms them.

---

## Architecture

```
safety (deterministic, FIRST, cannot be bypassed)
  -> understand + plan          api/understand.py   (Claude OR Gemma via api/llm.py)
  -> domain gate                in_domain=false -> out_of_domain, no retrieval
  -> verse lookup               api/verse.py        (deterministic citation match)
  -> named-source preference    api/verse.detect_sources
  -> school grounding limit     api/schools.py
  -> creative detection         api/creative.py     (form + output language)
  -> retrieve                   api/retrieve.py     (BGE-M3 + tradition filter + rerank)
  -> generate                   api/generate.py     (persona per mode, [P#] grounded)
  -> verify                     api/verify.py       (deterministic; §19 for creative)
  -> memory                     api/memory.py       (short-term always; long-term gated)
```

**Two shared entry points — do not bypass them.** `pipeline.prepare()` owns routing and
retrieval; `pipeline.generate_reply()` owns every generation-time decision (creative guard,
verse guards, faithfulness guard, streaming, `temperature`). The eval calls both. This rule
exists because the eval twice kept a private copy of a pipeline stage and a whole run
measured code the product does not execute.

### Modes

`counseling` · `teaching` · `verse` · `creative` · `out_of_domain` · `shastrarth` (parked,
no data). Routing is decided by `understand()` except verse and creative, which are
detected deterministically.

---

## Run it

```bash
# V2 — the from-scratch Gemma, no API key needed anywhere (GPU, ~10 min cold load)
bash serve.sh

# V1 — Claude backend (needs ANTHROPIC_API_KEY)
uvicorn api.main:app --port 8000

# deploy gate, deterministic only, zero API cost
python -m eval.six_gate --backend gemma --judge none --k 3 --out eval/run.json
python -m eval.watch_gates --out eval/run.json --total 297 --fails   # live progress

# rebuild the served index after any KB or config change
python -m api.build_index

# tests (pytest is NOT installed — run modules directly)
python -m api.tests.test_safety_memory
python -m api.tests.test_verse_grammar_guard
python -m api.tests.test_verse_pipeline_guard
python -m api.tests.test_named_source_and_schools
python -m api.tests.test_verify_chapter_verse
```

Long GPU jobs go through `run_deploy.sh` / `run_quant.sh` — detached under `nohup`,
sequential (one 52 GB model at a time), every stage skipped if its output exists.

### Environment

| var | default | meaning |
|---|---|---|
| `SATSANG_GEN_BACKEND` | `claude` | `gemma` = the tuned adapter writes the reply |
| `SATSANG_UTILITY_BACKEND` | `claude` | `gemma` = planning/extraction too; both gemma = Claude-free |
| `SATSANG_GEMMA_ADAPTER` | `v2/data/gemma4-v2-dpo2-lora` | base + this LoRA |
| `SATSANG_GEMMA_MODEL` | *(unset)* | a STANDALONE merged/quantized dir; overrides the adapter |
| `SATSANG_SHASTRARTH` | off | offer shastrarth as a client-selectable mode |
| `SATSANG_STORE` | `memory` | `postgres` for pgvector |
| `SATSANG_RERANK` | on | cross-encoder reranking |

---

## V2 (the from-scratch model)

Base **Gemma 4 26B MoE** (`google/gemma-4-26B-A4B-it`, 3.8B active/token) → QLoRA **SFT** →
**DPO**. Shipped adapter: `v2/data/gemma4-v2-dpo2-lora`.

* SFT: 447 steps / 3 epochs, train loss 3.14 → 0.815, eval 1.057 → 0.908.
* DPO v1 was **invalid** — rewards/accuracies 1.0 from step 50, because `rejected` was the
  chosen reply plus one of ~3 canned strings per flaw. The model learned "boilerplate = bad".
* DPO v2 fixed it by sampling `rejected` **from the SFT model itself**
  (`v2/onpolicy_negatives.py`): accuracies 0.41 → 0.98, a healthy curve.
* Both DPO runs show `rewards/chosen` going negative (~−2.6) = likelihood displacement.
  Safe here only because `api/verify` checks every `[P#]` deterministically regardless.

**Three attempts to beat dpo2 all failed** — 6-passage contexts (lost 6-2 in blinded
pairwise), naturalized problems, and a bilingual retrain (the premise was false: dpo2 was
already fluent Gujarati, 0.98 script ratio, having never been trained on it). Synthetic data
has plateaued. Only real user conversations will move it.

### 🚨 Quantization: bitsandbytes cannot quantize this MoE

The "4-bit" model came out **46 GB**, the "8-bit" **47 GB**, versus 49 GB bf16. Reading
tensor dtypes from the safetensors header:

```
BF16   47.20 GB   <- untouched
U8      1.14 GB   <- actually quantized   => 2.4% of weights
```

`transformers` stores Gemma-4 experts as **fused 3-D parameter tensors**
(`layers.N.experts.gate_up_proj`, `.down_proj`), and bitsandbytes only replaces
`nn.Linear`. It quantized the attention projections and left every expert in bf16.
`load_in_4bit: True` was obeyed; it had almost nothing to act on. This also explains the
+6.92% perplexity — real damage to attention for ~zero memory saving.

**The 4090 economics are unreachable on this path.** Revisiting needs MoE-aware
quantization (AWQ/GPTQ, llm-compressor/compressed-tensors) *plus* verified support for
fused experts. `awq` and `vllm` are not installed here.

**`du -sh` the output of any quantization before measuring it.** A "4-bit" model the size
of bf16 is quantization that did not happen.

---

## Evaluation

`eval/six_gate.py` is the deploy gate (proposal §20.3). `--judge {opus,sonnet,none}`;
`none` is deterministic-only at zero cost and is the default way to work.

* **Always `--k 3` for a decision.** Measured run-to-run noise on identical inputs:
  hallucination ±0.13 — larger than every treatment effect this project has measured.
* **Verdicts are segmented by mode and all of them are persisted.** A combined average once
  read REJECT while counseling passed all six; conversely `counseling_deploy: true` once
  masked verse, teaching, creative and out_of_domain all failing.
* `--backend {env,claude,gemma}` genuinely selects the runtime. It used to be decorative —
  parsed, defaulted to `claude`, never read — which is how a k=3 run was filed as a deploy
  gate for a model it never touched.
* Resume is **backend-scoped**: a gemma run will not reuse replies from a claude sidecar.
* `eval/rescore.py` re-derives verdicts from saved replies when a deterministic detector
  changes — free. It does **not** re-run `verify`, deliberately: reconstructing passages
  from the saved block loses passage text, and `verify_creative` needs it (attempting it
  reported creative 1.000 → 0.667, entirely fictional).
* `v2/quant_eval.py` compares two models: completion-only perplexity on the real held-out
  split, plus greedy divergence at temperature 0.

---

## Hard-won lessons

- **Regex detectors are the #1 source of false signal here — six incidents.** `take (an?|\d)`
  matched "just take a breath"; `dosage` matched inside a refusal; `_PUSHBACK` missed
  "gentle"; a verse gate fired on its own honest disclaimer; `advaita` matched inside
  *shuddhadvaita*; `_LOOSE_REF` flagged "Bhagavad Gita, Chapter 2, Verse 20" as
  unverified when that exact verse was in context. **Always read the flagged text first.**
- **A saturated metric cannot detect a regression.** With every gate at 1.000 there is no
  headroom; measuring quantization damage needed perplexity and divergence, not the gates.
- **Measure the thing you think you are measuring.** Perplexity scored over the whole
  sequence — 71.7% of which is prompt the model was trained (via `completion_only_loss`)
  *not* to predict — gave ppl ~5200 and made 4-bit look 21% *better* than bf16. A confident,
  plausible number pointing the wrong way is more dangerous than a crash.
- **Guards teach evasion.** Blocked from a "word-by-word" heading, the model produced the
  same fabricated glosses under "Breaking Down the Three Key Words". Detect the *shape* of
  the thing, not its label.
- **An id list cannot survive a re-chunk.** 8 of 14 hand-listed exclusions silently unbound,
  putting a conference title page back into the served index. Exclusions are rules now.
- **Live probes beat unit tests** for this system — the crisis method-seeking gap
  ("how many paracetamol…") and both pipeline-drift bugs were found by running real turns.
- **CPU tests cannot catch runtime-only failures**: a `NameError` on an unexercised path, or
  two 52 GB models not fitting on one 80 GB card. Exercise the caller, not just the helper.
- **Prompting has limits.** Three creative-persona revisions all landed inside noise; a
  deterministic guard fixed it in one attempt.

## Environment gotchas

- GPU: H100 80 GB (Hopper `grouped_mm`). Blackwell needs `experts_implementation="eager"`.
- **One 52 GB model at a time.** A second load wedges for ~10 min then dies.
- Cold weight load from Lightning network storage: **~10–12 min**; warm ~9 s.
- `ANTHROPIC_API_KEY` lives in `~/.zshrc`; non-interactive shells don't source it — prefix
  with `source ~/.zshrc 2>/dev/null;`. **Credits are exhausted.**
- `pytest` is not installed; run test modules directly.
- Foreground jobs through `| tee` die silently on terminal disconnect — use the run scripts.

## What's left, by priority

1. **Ship bf16.** It passes. Stop optimising it.
2. **Morphology from a grammar engine** (Vidyut / sanskrit_parser / Heritage / DCS) — not
   from the LLM. 30,859 corpus rows already carry `word_meanings` to validate against.
3. **Smaller-model experiment** — the RAG carries the knowledge; the tuned model supplies
   persona and grounding. A smaller *dense* model also quantizes normally.
4. **Judged quality gates on V2** when API budget allows.
5. **Serve it to real users** — the only source of preference data that will beat dpo2.
6. Knowledge graph (§12.1) last: the bottleneck is coverage and morphology, not linking.
