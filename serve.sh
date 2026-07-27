#!/usr/bin/env bash
# Serve SatsangAI V2 — the from-scratch Gemma, with NO Anthropic dependency.
#
# Both backends are set deliberately. With only GEN_BACKEND switched, every turn still
# made two Claude calls (understand+plan, fact extraction) and the server could not start
# without an API key — the model was tuned but the runtime was not free. UTILITY_BACKEND
# reuses the SAME 52 GB weights with the LoRA adapter disabled, so this costs no extra
# VRAM: the base plans, the tuned adapter speaks as the saint.
#
# First request loads ~52 GB and takes 3-6 minutes; every request after is warm.
set -euo pipefail
cd "$(dirname "$0")"

export SATSANG_GEN_BACKEND=gemma
export SATSANG_UTILITY_BACKEND=gemma
export SATSANG_EMBED_DEVICE=${SATSANG_EMBED_DEVICE:-cuda}   # BGE-M3 + reranker, ~2.5 GB
export SATSANG_HELPLINES_VERIFIED=${SATSANG_HELPLINES_VERIFIED:-1}

# One 52 GB model fits on this card. A stale eval or a second server holding VRAM is the
# difference between "warm in 4 minutes" and "OOM after 20".
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "!! Something is already using the GPU:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  echo "!! Free it with: pkill -f 'local_smoke|six_gate|uvicorn'"
  exit 1
fi

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
