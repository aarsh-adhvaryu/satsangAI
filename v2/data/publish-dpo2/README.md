---
license: gemma
base_model: google/gemma-4-26B-A4B-it
library_name: peft
tags: [lora, dpo, spiritual, gujarati, sanskrit, rag]
---

# SatsangAI — Gemma 4 26B saint-companion adapter (DPO)

A LoRA adapter that turns `google/gemma-4-26B-A4B-it` into a warm, grounded
saint-companion for the Swaminarayan (Akshar-Purushottam) and broader Hindu traditions.
It answers real personal problems through scripture — problem-first, citing only passages
it was given, and pushing back lovingly rather than flattering.

**This adapter is one component of a system, not a chatbot on its own.** It is trained to
answer from passages supplied in the prompt. Served without retrieval and without the
deterministic citation verifier it will behave like any other model asked to sound
scriptural — see "Intended use" below.

## Training

- **Base**: `google/gemma-4-26B-A4B-it` (26B MoE, ~4B active), bf16 LoRA r64/α128, all
  attention + MLP projections.
- **Stage 1 — SFT**: ~5,000 scripture-derived preference pairs, one grounding passage per
  example, completion-only loss.
- **Stage 2 — DPO**: on-policy negatives sampled from the SFT model itself (80%) plus
  rule-based flaw injection (20%). β 0.1, lr 5e-6, 1 epoch.

Training data was generated offline from a private corpus of Hindu and Swaminarayan
scripture. The runtime is Claude-free; a Claude model was used only to bootstrap offline
training data, never at inference.

## Results

Deterministic gates over 99 adversarial probes, served through the full SatsangAI pipeline
with **no external API**:

| mode | n | hallucination |
|---|---|---|
| counseling | 41 | 1.000 |
| verse explanation | 12 | 1.000 |
| creative writing | 12 | 1.000 |
| out-of-domain refusal | 4 | 1.000 |
| teaching | 30 | 0.967 |

Mode routing 34/34. Zero attribution-contract breaches.

Blinded pairwise judging preferred this DPO adapter over its SFT parent on
anti-sycophancy ("loving pushback") 4–0, never worse.

**Known limits.** ~42% of creative requests are refused rather than answered, because the
attribution guard blocks any line that presents invented words as scripture — safe, but
not yet good. Comparative multi-school ("shastrarth") answers are weak: that corpus is
unenriched OCR. The model is fluent in Gujarati and Hinglish as well as English.

## Intended use

Built for the SatsangAI pipeline: deterministic crisis screening → retrieval over a
tradition-filtered scripture index → generation grounded in the retrieved passages →
deterministic citation verification.

**Not intended** as a standalone chatbot, a source of religious authority, a therapist, or
a medical/legal adviser. Serving it without the crisis classifier and the citation verifier
removes the safety and grounding properties it was built for.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "google/gemma-4-26B-A4B-it"
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "aarsh-adhvaryu/satsangai-gemma4-dpo2")
```

~52 GB in bf16. Code: https://github.com/aarsh-adhvaryu/satsangAI
