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

## Gated next steps (need a GPU session — ASK the owner to enable it, bundle them)

1. **`chosen` generation.** Either `--chosen claude` (Opus, OFFLINE gold only —
   bootstrap; ~like the enrichment gold) or, for strict purity, `--chosen gemma`
   after SFT (the model self-generates, filtered by `api/verify` + a quality gate).
   Optionally have the teacher also paraphrase the mined problems for higher fidelity.
2. **SFT (QLoRA)** the base Gemma on the `chosen` replies — reuse
   `enrichment/qlora_train.py` (4-bit bnb, LoRA on `language_model` `Linear4bit`,
   `experts_implementation="eager"` for Blackwell).
3. **DPO** on the pairs (`trl` DPOTrainer over the same LoRA target set), every batch
   carrying the anti-drift negatives.
4. **6-gate eval (§20.3)** before any deploy — extend `eval/` (hallucination,
   persona, sycophancy, emotional-appropriateness, scripture-accuracy, RAGAS).
