"""Step 1 of V2 alignment — QLoRA **SFT** of Gemma-4 26B MoE on the faithful
`chosen` saint replies. Establishes the persona + grounding behaviour that DPO
then sharpens. Completion-only loss (prompt masked). GPU-only; run nothing until
the owner enables the GPU.

    python -m v2.sft_train --data v2/data/pairs.jsonl --epochs 3
    python -m v2.sft_train --smoke        # de-risk the stack (few steps, tiny data)

Accuracy config lives in v2/train_config.py (rank 32, all attn+MLP, bf16 NF4,
eval split). See its docstring for the QLoRA tradeoff reasoning.
"""
from __future__ import annotations

import argparse
import os

from v2 import train_config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.PAIRS))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)     # SFT: standard LoRA LR
    ap.add_argument("--precision", default="bf16", choices=["bf16", "4bit"],
                    help="bf16 = max quality (default); 4bit = QLoRA OOM fallback")
    ap.add_argument("--bs", type=int, default=2, help="per-device micro-batch — raise to fill VRAM")
    ap.add_argument("--ga", type=int, default=16, help="grad-accum (effective batch = bs*ga)")
    ap.add_argument("--out", default=str(C.SFT_OUT))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    from peft import get_peft_model
    from trl import SFTConfig, SFTTrainer

    # Smoke runs write to a throwaway dir so they never leave a junk 6-step adapter
    # (or stale checkpoints) in the real --out that the DPO stage would then load.
    out = f"{a.out}-smoke" if a.smoke else a.out

    C.tune_runtime()                                       # TF32 + report GPU/MoE kernel
    model, tok = C.load_base(a.precision)
    model = C.prepare_for_training(model, a.precision)
    model = get_peft_model(model, C.lora_config(model))
    model.print_trainable_parameters()

    train_ds, eval_ds = C.sft_dataset(tok, a.data)
    if a.smoke:
        train_ds, eval_ds = train_ds.select(range(min(8, len(train_ds)))), \
                            eval_ds.select(range(min(2, len(eval_ds))))
    print(f"SFT on {len(train_ds)} train / {len(eval_ds)} eval examples")

    cfg = SFTConfig(
        output_dir=out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.ga,
        learning_rate=a.lr, warmup_ratio=0.03, lr_scheduler_type="cosine",
        bf16=True, max_length=C.MAX_LEN, completion_only_loss=True, packing=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, eval_strategy="steps", eval_steps=25,
        save_strategy="steps", save_steps=25, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", max_steps=6 if a.smoke else -1)

    trainer = SFTTrainer(model=model, args=cfg,
                         train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tok)
    # Auto-resume: if a prior run left checkpoints in --out, continue from the last
    # one (restores optimizer/scheduler/RNG/step). Re-run the same command to resume.
    from transformers.trainer_utils import get_last_checkpoint
    ckpt = get_last_checkpoint(out) if (not a.smoke and os.path.isdir(out)) else None
    if ckpt:
        print(f"resuming SFT from checkpoint: {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f"saved SFT LoRA adapter -> {out}")


if __name__ == "__main__":
    main()
