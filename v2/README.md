# V2 — Claude-free generation Gemma (SFT + DPO alignment)

V2's goal (proposal Part II §20): align the **from-scratch generation model**
(Gemma 4 26B MoE) to hold the saint-companion persona — warm, deep, **lovingly
disagrees** (anti-sycophancy), zero-hallucination — **without Claude at runtime**,
fed increasingly by V1's own conversation data.

Order: **SFT (QLoRA) first**, then **DPO** on preference pairs. Enrichment used
SFT only; DPO is specifically for persona/alignment (proposal §20.2).

## Status

**Data pipeline scaffolded (CPU, no GPU, no API) — this directory.** The
preference-pair schema, the scripture-derived seed miner, and the Claude-free
`rejected`-side flaw injectors all run and are unit-tested now. Training and the
high-quality `chosen` generation are the GPU/decision-gated steps below.

## Decisions (locked, reversible — mirror the enrichment track)

- **Chosen-source = Hybrid.** Bootstrap DPO pairs offline now; progressively
  replace with real V1 conversation pairs as they accumulate (only ~26 turns
  exist today — nowhere near enough yet). Runtime stays Gemma-only.
- **`rejected` side is fully Claude-free** — deterministic flaw injection, so the
  negatives never depend on any model. The `chosen` side is the only place a
  bootstrap teacher (Claude Opus, offline) would enter; kept pluggable so a
  strict never-Claude regeneration stays possible (source untouched).

## DPO data = three sources (§20.2)

1. **Scripture-derived (~5,000+)** — built here. Seeds are mined from the enriched
   core's `when_this_helps` (the situations each passage addresses) → first-person
   problems, grounded in that passage. `v2/seeds.py`.
2. **Real V1 conversations** — later; harvested from the Postgres/JSON conversation
   store once traffic exists.
3. **Synthetic hard cases** — targeted adversarial/persona edges (extend `v2/reject.py`).

## Anti-drift coverage (§20.4)

The `rejected` taxonomy in `v2/reject.py` *is* the anti-drift battery — every batch
carries these negatives:

| flaw | rule it violates | anti-drift dimension |
|------|------------------|----------------------|
| `hallucinated_citation` | zero-hallucination | hallucination creep |
| `sycophancy` | loving pushback, no flattery | sycophancy |
| `shallow` | depth, no platitudes | depth erosion |
| `doctrine_mix` | never mix schools in counseling | theological drift |
| `off_tradition` | stay in-tradition | cross-contamination |
| `name_fabrication` | no names/dates not in passages | over-attribution |

## Files

- `schema.py` — `PreferencePair` (+ jsonl io); `render_prompt()` mirrors V1's real
  generation user-turn (`api/generate.py`) so SFT/DPO train on the served distribution.
- `seeds.py` — mine `when_this_helps` → stratified first-person problem seeds.
- `reject.py` — Claude-free flaw injectors → labelled negatives.
- `build_pairs.py` — orchestrator; `--chosen {placeholder|claude|gemma}`.
- `tests/test_pipeline.py` — CPU smoke tests (no model/API).

## Run

```bash
# CPU, now — smoke-test the whole pipeline with a grounded template `chosen`:
python -m v2.build_pairs --n 200 --chosen placeholder --out v2/data/pairs_sample.jsonl
python -c "import v2.tests.test_pipeline as t; [getattr(t,f)() for f in dir(t) if f.startswith('test_')]"
```

## Tuning stack (written, accuracy-tuned; GPU-gated — run NOTHING until owner enables GPU)

The tuners are staged and compile-checked; the CPU data-prep path is verified. They
only load the model inside `main()`, so nothing touches CUDA until you launch them.

**Max quality on H100 / RTX PRO 6000.** With 80–96 GB we don't accept the QLoRA
accuracy tradeoff at all: the default is **full-precision bf16 LoRA (no 4-bit)** —
26B in bf16 is ~52 GB, which fits — so there's **no quantization error**. `--precision
4bit` remains only as an OOM fallback. Capacity is high (**LoRA rank 64 / alpha 128**,
all attn+MLP). The config is **hardware-aware**: on **H100 (Hopper)** it uses the fast
`grouped_mm` MoE kernel; on **Blackwell** it auto-switches to `eager` (torch 2.8's
grouped_mm is Hopper-only). TF32 + SDPA on. Raise `--bs` to fill the card (`--smoke`
first to find the max micro-batch; 96 GB Blackwell fits more than 80 GB H100).

- `train_config.py` — shared config + the **precision / accuracy writeup** (bf16-first,
  rank 64, hardware-aware MoE kernel, eval split, conservative DPO knobs). Read before running.
- `sft_train.py` — **Step 1: QLoRA SFT** on the faithful `chosen` replies
  (completion-only loss, eval split, best-checkpoint). Establishes persona + grounding.
- `dpo_train.py` — **Step 2: QLoRA DPO** on the pairs, continuing from the SFT
  adapter; reference = SFT policy with adapter disabled; LR 5e-6 / 1 epoch / beta 0.1
  (DPO overfits fast on a 4-bit policy and can reward-hack length).

### Gated run order (bundle into one GPU session)

```bash
# 0. generate real `chosen` (offline) and write the real pairs file:
python -m v2.build_pairs --n 5000 --chosen claude --out v2/data/pairs.jsonl   # OR --chosen gemma
# 1. de-risk the stack + find the max micro-batch that fits (few steps, tiny data):
python -m v2.sft_train --smoke --bs 4   &&   python -m v2.dpo_train --smoke --bs 2
# 2. real runs (bf16 max-quality by default; push --bs up to fill the GPU):
python -m v2.sft_train --data v2/data/pairs.jsonl --epochs 3 --bs 4 --ga 8
python -m v2.dpo_train --data v2/data/pairs.jsonl --sft v2/data/gemma4-v2-sft-lora --bs 2 --ga 8
# 3. 6-gate eval (§20.3) before any deploy — extend eval/:
#    hallucination · persona · sycophancy · emotional-appropriateness · scripture-accuracy · RAGAS
```

### Watch-items for the GPU run
- **`chosen` source first** — pick `--chosen claude` (bootstrap, offline gold) or the
  strict-pure `--chosen gemma` (needs an initial SFT to self-generate). Optionally have
  the teacher paraphrase the mined problems for higher fidelity than the regex seeds.
- **Data-quality gate** — before DPO, drop any pair whose `chosen` fails `api/verify`
  (don't let DPO reinforce an ungrounded chosen). Add as a `build_pairs` filter.
- **Gemma-4 "thought" channel** — the chat template opens a `thought` channel at the
  generation point. The enrichment tuner used this exact template with a plain
  completion and trained cleanly, so the tuners follow that proven pattern; if you want
  explicit chain-of-thought before the saint reply, format `chosen` with a thought segment.
- **`pytest` is not installed** — run the smoke tests via the dir-loop one-liner above.
