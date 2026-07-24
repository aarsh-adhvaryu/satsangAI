"""Production inference for the V2 Gemma adapter via vLLM (proposal §21).

The FastAPI app (SATSANG_GEN_BACKEND=gemma) loads the 52 GB bf16 model in-process — fine
for one box, wasteful at scale. vLLM gives continuous batching + paged-KV so one GPU serves
many concurrent users. This script merges the LoRA into the base once, then launches vLLM's
OpenAI-compatible server on the merged weights.

    # 1. merge adapter -> standalone model (GPU, ~52 GB out):
    python -m v2.serve_vllm merge --adapter v2/data/gemma4-v2-dpo2-lora --out v2/data/gemma4-v2-merged
    # 2. serve it (GPU):
    python -m v2.serve_vllm serve --model v2/data/gemma4-v2-merged --port 8001

Caveat: vLLM must support the gemma4 MoE arch (needs a recent vLLM with the transformers
fallback backend + experts_implementation). If vLLM can't load it, keep the in-process
FastAPI backend (SATSANG_GEN_BACKEND=gemma) — correctness is identical, only throughput differs.
For a smaller/faster server, quantize the merged model to AWQ/GPTQ 4-bit first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def merge(adapter: str, out: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from v2.train_config import MODEL, detect_gpu

    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if not detect_gpu()["is_hopper"]:
        kw["experts_implementation"] = "eager"
    base = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(out)
    print(f"merged {adapter} into {MODEL} -> {out}")


def serve(model: str, port: int, extra: list[str]) -> None:
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model, "--port", str(port),
           "--served-model-name", "satsangai-v2",
           "--dtype", "bfloat16", "--max-model-len", "4096", *extra]
    print("launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge"); m.add_argument("--adapter", required=True); m.add_argument("--out", required=True)
    s = sub.add_parser("serve"); s.add_argument("--model", required=True); s.add_argument("--port", type=int, default=8001)
    s.add_argument("extra", nargs="*", help="extra vLLM flags, e.g. --quantization awq")
    a = ap.parse_args()
    if a.cmd == "merge":
        merge(a.adapter, a.out)
    else:
        serve(a.model, a.port, a.extra)


if __name__ == "__main__":
    main()
