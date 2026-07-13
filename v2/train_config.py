"""Shared QLoRA config + data prep for the V2 SFT and DPO tuners.

QLoRA accuracy tradeoff — and how this config manages it
--------------------------------------------------------
QLoRA freezes the base in 4-bit NF4 and learns low-rank adapters. The 4-bit
quantization of the frozen weights introduces error the adapters must compensate
for, so a careless config loses accuracy vs full/bf16 fine-tuning. Because this
project is "quality over speed/cost", we spend capacity to close that gap:

  * NF4 + double-quantization + **bf16 compute** dtype (not fp16) — the QLoRA-paper
    recipe; double-quant recovers memory so we can afford a higher rank.
  * **Rank 32 / alpha 64** (2x the enrichment task's r16). Generation-persona is a
    harder target than the structured enrichment task, and higher rank measurably
    narrows the QLoRA↔full-FT gap. Adapt **all** attention + MLP projections
    (q,k,v,o,gate,up,down) — broad target coverage matters more than depth.
  * Leave the **MoE router and the vision tower unadapted** (router adaptation is
    unstable; PEFT can't wrap the vision `Gemma4ClippableLinear`).
  * **Eval split + best-checkpoint selection** so we detect the accuracy/overfit
    knee instead of trusting train loss.
  * DPO uses a **much lower LR (5e-6), 1 epoch, moderate beta** — DPO on a QLoRA
    policy overfits fast and can reward-hack length/verbosity; the SFT model is the
    frozen reference (via adapter-disable) to anchor faithfulness.

Blackwell note: Gemma-4 MoE must load with experts_implementation="eager" (torch
2.8's grouped_mm MoE kernel is Hopper-only), for both train and inference.
"""
from __future__ import annotations

from pathlib import Path

MODEL = "google/gemma-4-26B-A4B-it"
DATA = Path(__file__).resolve().parent / "data"
PAIRS = DATA / "pairs.jsonl"                 # produced by v2/build_pairs.py
SFT_OUT = DATA / "gemma4-v2-sft-lora"
DPO_OUT = DATA / "gemma4-v2-dpo-lora"

# LoRA (accuracy-oriented — see module docstring)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj")

MAX_LEN = 1536          # grounded prompt (persona + passages) + reply
EVAL_FRACTION = 0.05    # held-out split to watch the accuracy/overfit knee


def bnb_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)


def load_base():
    """4-bit Gemma-4 MoE on CUDA, eager experts (Blackwell). GPU-only."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb_config(), dtype=torch.bfloat16,
        device_map="cuda", experts_implementation="eager")
    model.config.use_cache = False
    return model, tok


def lora_targets(model) -> list[str]:
    """Full paths of the text-model Linear4bit attn+MLP projections to adapt."""
    return [n for n, m in model.named_modules()
            if type(m).__name__ == "Linear4bit"
            and "language_model" in n
            and n.split(".")[-1] in LORA_TARGET_SUFFIXES]


def lora_config(model):
    from peft import LoraConfig
    return LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=lora_targets(model))


def _persona_prompt(problem: str, context: str) -> str:
    """The saint persona + the grounded user-turn — matches what V1 serves."""
    from api.generate import PERSONA
    user = (f"The person wrote:\n\"{problem}\"\n\n"
            f"PASSAGES (cite only these, by tag):\n{context}\n\n"
            f"Respond to the person now as the saint-companion.")
    return PERSONA + "\n\n" + user


def render_prompt(tok, problem: str, context: str) -> str:
    """Chat-templated prompt string ending at the assistant generation point."""
    return tok.apply_chat_template(
        [{"role": "user", "content": _persona_prompt(problem, context)}],
        add_generation_prompt=True, tokenize=False)


def _split(rows: list[dict], seed: int = 0):
    import random
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    k = max(1, int(len(rows) * EVAL_FRACTION))
    return rows[k:], rows[:k]                    # train, eval


def sft_dataset(tok, path: str | Path):
    """(prompt, completion) — completion-only loss masks the prompt. For SFT on the
    faithful `chosen` replies."""
    from datasets import Dataset
    from v2.schema import read_jsonl
    rows = [{"prompt": render_prompt(tok, p.problem, p.context),
             "completion": p.chosen.strip() + tok.eos_token}
            for p in read_jsonl(path)]
    tr, ev = _split(rows)
    return Dataset.from_list(tr), Dataset.from_list(ev)


def dpo_dataset(tok, path: str | Path):
    """(prompt, chosen, rejected) for DPO — same grounded prompt distribution."""
    from datasets import Dataset
    from v2.schema import read_jsonl
    rows = [{"prompt": render_prompt(tok, p.problem, p.context),
             "chosen": p.chosen.strip(), "rejected": p.rejected.strip()}
            for p in read_jsonl(path)]
    tr, ev = _split(rows)
    return Dataset.from_list(tr), Dataset.from_list(ev)
