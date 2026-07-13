"""Step 2 of V2 alignment — QLoRA **DPO** on the preference pairs, continuing from
the SFT adapter. Teaches the model to prefer the faithful `chosen` over the
flaw-injected `rejected` (hallucination / sycophancy / shallowness / doctrine-mix
/ off-tradition / name-fabrication) — the anti-drift battery (proposal §20.4).

Reference model = the SFT policy with its adapter disabled (memory-free implicit
reference, standard for QLoRA-DPO). GPU-only; run nothing until the owner enables
the GPU.

    python -m v2.dpo_train --data v2/data/pairs.jsonl --sft v2/data/gemma4-v2-sft-lora
    python -m v2.dpo_train --smoke

Why the conservative knobs (accuracy tradeoff): DPO on a 4-bit QLoRA policy
overfits quickly and can reward-hack verbosity; so LR is 5e-6, 1 epoch, beta 0.1,
and we watch eval reward-accuracy / margins on a held-out split.
"""
from __future__ import annotations

import argparse

from v2 import train_config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.PAIRS))
    ap.add_argument("--sft", default=str(C.SFT_OUT), help="SFT LoRA adapter to continue from")
    ap.add_argument("--epochs", type=float, default=1)     # DPO overfits fast
    ap.add_argument("--lr", type=float, default=5e-6)      # DPO: ~40x lower than SFT
    ap.add_argument("--beta", type=float, default=0.1)     # KL strength to the SFT ref
    ap.add_argument("--out", default=str(C.DPO_OUT))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    base, tok = C.load_base()
    # Load the SFT adapter as the trainable policy; DPOTrainer uses the same adapter
    # DISABLED as the frozen reference (ref_model=None + a PEFT policy).
    model = PeftModel.from_pretrained(base, a.sft, is_trainable=True)

    train_ds, eval_ds = C.dpo_dataset(tok, a.data)
    if a.smoke:
        train_ds, eval_ds = train_ds.select(range(min(8, len(train_ds)))), \
                            eval_ds.select(range(min(2, len(eval_ds))))
    print(f"DPO on {len(train_ds)} train / {len(eval_ds)} eval pairs | beta={a.beta}")

    cfg = DPOConfig(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,   # eff. batch 16
        learning_rate=a.lr, warmup_ratio=0.1, lr_scheduler_type="cosine",
        beta=a.beta, loss_type="sigmoid", max_length=C.MAX_LEN,
        precompute_ref_log_probs=True,                                   # ref = adapter-disabled
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5, eval_strategy="steps", eval_steps=25,
        save_strategy="steps", save_steps=25, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_rewards/accuracies",
        greater_is_better=True,
        report_to="none", max_steps=6 if a.smoke else -1)

    trainer = DPOTrainer(model=model, ref_model=None, args=cfg,
                         train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print(f"saved DPO LoRA adapter -> {a.out}")


if __name__ == "__main__":
    main()
