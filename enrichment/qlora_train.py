"""QLoRA fine-tune Gemma 4 26B MoE on the Claude-distilled gold set.

4-bit base (bitsandbytes) + LoRA adapters. Completion-only loss: the prompt tokens
are masked (-100), the model is trained only on the JSON enrichment. Forces the
eager MoE experts path (Blackwell can't use torch 2.8's grouped_mm kernel).

    # real run (after gold.jsonl exists):
    python -m enrichment.qlora_train --epochs 3
    # de-risk the training stack on Blackwell without Claude:
    python -m enrichment.qlora_train --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

from .prompt import SYSTEM, build_prompt, FIELDS

MODEL = "google/gemma-4-26B-A4B-it"
SAMPLE = Path("enrichment/data/gold_seed_sample.parquet")
GOLD = Path("enrichment/data/gold.jsonl")
OUT = Path("enrichment/data/gemma4-enrich-lora")
MAXLEN = 1024


def load_examples(smoke: bool) -> list[dict]:
    """Join gold.jsonl onto the sampled rows to rebuild (prompt, completion) pairs."""
    df = pd.read_parquet(SAMPLE).set_index("id")
    if smoke:
        rows = df.head(6)
        gold = {i: {"contextual_explanation": "Placeholder explanation for smoke test.",
                    "when_this_helps": "When validating the training stack.",
                    "core_principle": "Smoke test only",
                    "gujarati_explanation": "પરીક્ષણ માટે."} for i in rows.index}
    else:
        gold = {r["id"]: r for r in map(json.loads, GOLD.read_text().splitlines())}
        rows = df.loc[[i for i in gold if i in df.index]]
    out = []
    for rid, row in rows.iterrows():
        g = gold[rid]
        completion = json.dumps({k: g.get(k) for k in FIELDS if g.get(k)},
                                ensure_ascii=False)
        out.append({"prompt": SYSTEM + "\n\n" + build_prompt(row.to_dict()),
                    "completion": completion})
    return out


def tokenize(ex: dict, tok) -> dict:
    """Mask prompt tokens; train only on the assistant JSON."""
    p_ids = tok.apply_chat_template(
        [{"role": "user", "content": ex["prompt"]}],
        add_generation_prompt=True, tokenize=True, return_dict=False)
    c_ids = tok(ex["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + c_ids)[:MAXLEN]
    labels = ([-100] * len(p_ids) + c_ids)[:MAXLEN]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def lora_targets(model) -> list[str]:
    """Full paths of the text-model Linear4bit projections to adapt. Restricted to
    `language_model` (skip the vision tower, whose Gemma4ClippableLinear wrappers
    PEFT can't adapt) and to attention + MLP projections (skip the MoE router)."""
    keep = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return [n for n, m in model.named_modules()
            if type(m).__name__ == "Linear4bit"
            and "language_model" in n and n.split(".")[-1] in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, dtype=torch.bfloat16,
        device_map="cuda", experts_implementation="eager")
    model.config.use_cache = False

    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=lora_targets(model))
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    exs = load_examples(args.smoke)
    ds = Dataset.from_list(exs).map(lambda e: tokenize(e, tok),
                                    remove_columns=["prompt", "completion"])
    print(f"training on {len(ds)} examples | LoRA targets: {lcfg.target_modules}")

    targs = TrainingArguments(
        output_dir=str(OUT), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=args.lr, warmup_ratio=0.03, lr_scheduler_type="cosine",
        bf16=True, logging_steps=5, save_strategy="epoch",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", max_steps=8 if args.smoke else -1)
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True))
    trainer.train()
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"saved LoRA adapter -> {OUT}")


if __name__ == "__main__":
    main()
