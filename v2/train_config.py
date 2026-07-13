"""Shared config + data prep for the V2 SFT and DPO tuners — tuned for MAX quality
and full utilisation of an H100 (80 GB) or RTX PRO 6000 Blackwell (96 GB).

Precision — eliminating the QLoRA accuracy tradeoff
---------------------------------------------------
QLoRA's tradeoff is the 4-bit NF4 freeze of the base: quantization error the
adapters must absorb. With 80-96 GB we don't have to accept it. The 26B MoE in
**bf16 is ~52 GB**, which fits — so the DEFAULT here is **full-precision bf16 LoRA
(no quantization)**: highest fidelity, "perfect-model" path. `--precision 4bit`
stays as an OOM fallback (NF4 + double-quant + bf16 compute).

Capacity: **LoRA rank 64 / alpha 128** over ALL attention + MLP projections
(q,k,v,o,gate,up,down) of the `language_model` — big VRAM lets us use the high
rank that closes the LoRA↔full-FT gap. MoE router + vision tower stay unadapted.

Hardware-aware MoE kernel (throughput)
--------------------------------------
Gemma-4 MoE: torch 2.8's fast `grouped_mm` experts kernel is **Hopper-only
(cc==9.0)**. So on **H100 we use the default fast kernel**; on **Blackwell
(cc>=10) we must pass experts_implementation="eager"**. Detected automatically.
We also enable TF32 matmuls and SDPA attention for max throughput, and size the
micro-batch to actually fill the card (bf16 base leaves plenty of headroom).
"""
from __future__ import annotations

from pathlib import Path

MODEL = "google/gemma-4-26B-A4B-it"
DATA = Path(__file__).resolve().parent / "data"
PAIRS = DATA / "pairs.jsonl"                 # produced by v2/build_pairs.py
SFT_OUT = DATA / "gemma4-v2-sft-lora"
DPO_OUT = DATA / "gemma4-v2-dpo-lora"

# LoRA — high capacity for a faithful persona (see docstring)
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj")

MAX_LEN = 2048          # grounded prompt (persona + passages) + reply, with headroom
EVAL_FRACTION = 0.05    # held-out split to watch the accuracy/overfit knee


# --------------------------------------------------------------------------- #
#  Hardware detection + runtime tuning (GPU-only; safe to import on CPU)       #
# --------------------------------------------------------------------------- #
def detect_gpu() -> dict:
    """{name, cc, is_hopper, is_blackwell, vram_gb} for the current CUDA device."""
    import torch
    if not torch.cuda.is_available():
        return {"name": "cpu", "cc": (0, 0), "is_hopper": False,
                "is_blackwell": False, "vram_gb": 0}
    cc = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "cc": cc,
            "is_hopper": cc[0] == 9,           # H100 == sm_90
            "is_blackwell": cc[0] >= 10,       # RTX PRO 6000 Blackwell == sm_12x
            "vram_gb": round(props.total_memory / 1e9, 1)}


def tune_runtime() -> dict:
    """TF32 matmul + report the GPU. Call once at the top of each tuner's main()."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gpu = detect_gpu()
    print(f"GPU: {gpu['name']} (cc {gpu['cc'][0]}.{gpu['cc'][1]}, {gpu['vram_gb']} GB) | "
          f"MoE kernel: {'grouped_mm (fast)' if gpu['is_hopper'] else 'eager'}")
    return gpu


def load_base(precision: str = "bf16"):
    """Load Gemma-4 MoE on CUDA at max quality. precision: 'bf16' (default, no
    quantization) or '4bit' (QLoRA fallback). Picks the MoE kernel by GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gpu = detect_gpu()
    kw = dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    if not gpu["is_hopper"]:                    # Blackwell (and non-Hopper) need eager MoE
        kw["experts_implementation"] = "eager"
    if precision == "4bit":
        kw["quantization_config"] = bnb_config()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
    model.config.use_cache = False
    return model, tok


def bnb_config():
    """QLoRA fallback quantization (only used with --precision 4bit)."""
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)


def prepare_for_training(model, precision: str):
    """Make the frozen base compatible with gradient-checkpointed LoRA training.
    Call after load_base(), before wrapping with LoRA."""
    if precision == "4bit":
        from peft import prepare_model_for_kbit_training
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()          # needed for checkpointing on a bf16 base
    return model


def lora_targets(model) -> list[str]:
    """Text-model attn+MLP projections to adapt (Linear4bit under 4bit, else Linear)."""
    keep = LORA_TARGET_SUFFIXES
    return [n for n, m in model.named_modules()
            if type(m).__name__ in ("Linear4bit", "Linear")
            and "language_model" in n and n.split(".")[-1] in keep]


def lora_config(model):
    from peft import LoraConfig
    return LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=lora_targets(model))


# --------------------------------------------------------------------------- #
#  Prompt + dataset prep (CPU-safe; needs only the tokenizer)                  #
# --------------------------------------------------------------------------- #
def _persona_prompt(problem: str, context: str) -> str:
    from api.generate import PERSONA
    user = (f"The person wrote:\n\"{problem}\"\n\n"
            f"PASSAGES (cite only these, by tag):\n{context}\n\n"
            f"Respond to the person now as the saint-companion.")
    return PERSONA + "\n\n" + user


def render_prompt(tok, problem: str, context: str) -> str:
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
    from datasets import Dataset
    from v2.schema import read_jsonl
    rows = [{"prompt": render_prompt(tok, p.problem, p.context),
             "completion": p.chosen.strip() + tok.eos_token}
            for p in read_jsonl(path)]
    tr, ev = _split(rows)
    return Dataset.from_list(tr), Dataset.from_list(ev)


def dpo_dataset(tok, path: str | Path):
    from datasets import Dataset
    from v2.schema import read_jsonl
    rows = [{"prompt": render_prompt(tok, p.problem, p.context),
             "chosen": p.chosen.strip(), "rejected": p.rejected.strip()}
            for p in read_jsonl(path)]
    tr, ev = _split(rows)
    return Dataset.from_list(tr), Dataset.from_list(ev)
