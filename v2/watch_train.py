"""Read-only progress monitor for the V2 training chain. Safe to run from any terminal.

Touches nothing the training job owns — reads checkpoint metadata, /proc and nvidia-smi. Use it
instead of watching the launching terminal: HF Trainer redraws a tqdm bar in place and only logs
every `logging_steps`, so a healthy run looks frozen, and if your terminal reconnects to a
different pts you will never see its output again.

    watch -n 60 python -m v2.watch_train

Liveness signal = the "ckpt Xm ago" field. save_steps=50 means a fresh checkpoint every ~10 min
during SFT; if that climbs past ~15 min while CPU stays high, then investigate.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = [("SFT", "v2/data/gemma4-v2-bi-sft-lora"), ("DPO", "v2/data/gemma4-v2-bi-dpo-lora")]
KEYS = ("sft_train", "dpo_train", "onpolicy_negatives", "multilingual_pairs",
        "eval_gates", "judge_pairwise", "six_gate", "shastrarth_translate")


def gpu() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def main() -> None:
    print(time.strftime("%H:%M:%S"), "| GPU:", gpu())

    ps = subprocess.run(["ps", "-eo", "pid,etime,pcpu,args"], capture_output=True, text=True).stdout
    live = [ln for ln in ps.splitlines()
            if any(k in ln for k in KEYS) and "grep" not in ln and "watch_train" not in ln]
    if live:
        for ln in live:
            f = ln.split()
            stage = next((k for k in KEYS if k in ln), "?")
            print(f"  RUNNING {stage:<20} pid={f[0]} elapsed={f[1]} cpu={f[2]}%")
    else:
        print("  no chain process found — either finished or died")

    # on-policy negatives has no checkpoint dir; its output file IS its resume state
    op = os.path.join(REPO, "v2/data/pairs_bilingual_onpolicy.jsonl")
    src = os.path.join(REPO, "v2/data/pairs_bilingual.jsonl")
    if os.path.exists(op):
        n = sum(1 for _ in open(op))
        tot = sum(1 for _ in open(src)) if os.path.exists(src) else 0
        pct = 100.0 * n / tot if tot else 0
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        age = int(time.time() - os.path.getmtime(op))
        print(f"  ONPOLICY: [{bar}] {n}/{tot} ({pct:.0f}%) | last write {age//60}m{age % 60}s ago")

    for label, path in STAGES:
        cks = sorted(glob.glob(os.path.join(REPO, path, "checkpoint-*")), key=os.path.getmtime)
        if not cks:
            print(f"  {label}: not started")
            continue
        try:
            s = json.load(open(os.path.join(cks[-1], "trainer_state.json")))
        except (OSError, json.JSONDecodeError):
            print(f"  {label}: checkpoint being written…")
            continue
        step, mx, ep = s["global_step"], s.get("max_steps") or 0, s.get("epoch", 0)
        age = int(time.time() - os.path.getmtime(cks[-1]))
        loss = next((x["loss"] for x in reversed(s["log_history"]) if "loss" in x), None)
        ev = next((x["eval_loss"] for x in reversed(s["log_history"]) if "eval_loss" in x), None)
        pct = (100.0 * step / mx) if mx else 0.0
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"  {label}: [{bar}] {step}/{mx} ({pct:.0f}%) epoch {ep:.2f} "
              f"loss={loss if loss is None else round(loss, 3)} "
              f"eval={ev if ev is None else round(ev, 4)} | ckpt {age//60}m{age % 60}s ago")

    # resumable eval sidecars
    for side in sorted(glob.glob(os.path.join(REPO, "eval/*.json.partial.jsonl"))):
        n = sum(1 for _ in open(side))
        print(f"  EVAL {os.path.basename(side):<34} {n} replies scored (resumable)")


if __name__ == "__main__":
    main()
