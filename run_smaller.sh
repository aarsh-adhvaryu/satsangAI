#!/usr/bin/env bash
# Train + evaluate a SMALLER base on the same data, trainers and gates as the 26B.
#
#   bash run_smaller.sh          # start (or resume)
#   bash run_smaller.sh status   # where is it
#   bash run_smaller.sh stop     # kill it
#   BASE=google/gemma-4-E4B-it bash run_smaller.sh     # a different candidate
#
# WHY: the shipped 26B is an MoE (enable_moe_block=True, 128 experts, top-8) whose experts
# are FUSED 3-D tensors. bitsandbytes only replaces nn.Linear, so it quantized 2.4% of the
# weights and the "4-bit" model came out 46 GB instead of 13 GB. The model is therefore
# stuck at 49 GB / H100 and the commodity-hardware economics are unreachable on that path.
#
# google/gemma-4-12B-it is DENSE (enable_moe_block=False), 23.9 GB, and quantizes normally
# to ~6-7 GB — which fits a 4090, an L4, or a 16 GB RTX 5060 Ti. It is also DEEPER and WIDER
# than the MoE: 48 layers / 3840 hidden vs 30 / 2816. Since the RAG supplies the knowledge
# and the tuned model supplies persona and grounding discipline, this is a genuine
# candidate, not a resigned downgrade.
#
# Everything is reused: the same 4,999 pairs, the same SFT/DPO trainers, the same 99-probe
# gate, and eval/six_gate_gemma_k3.json as the banked 26B baseline to compare against.
#
# Detached under nohup; every stage skipped if its output exists; re-run to resume.
set -uo pipefail
cd "$(dirname "$0")"

BASE="${BASE:-google/gemma-4-12B-it}"
TAG="$(basename "$BASE" | tr '.' '-')"
export SATSANG_BASE_MODEL="$BASE"

PAIRS=v2/data/pairs.jsonl
SFT=v2/data/${TAG}-sft-lora
ONPOLICY=v2/data/pairs_onpolicy_${TAG}.jsonl
DPO=v2/data/${TAG}-dpo-lora
GATE=eval/six_gate_${TAG}.json
MEASURE=v2/data/quant_${TAG}.json
MEASURE_PPL=v2/data/quant_${TAG}_ppl.json
MERGED=v2/data/${TAG}-merged
QUANT=v2/data/${TAG}-4bit
QGATE=eval/six_gate_${TAG}_4bit.json
LOG=eval/smaller_${TAG}.log
STATE=eval/.smaller_step_${TAG}

say() { echo "[$(date +%F' '%H:%M:%S)] $*"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0; }
wait_for_gpu() {
  local t=0
  while [ "$(gpu_used)" -gt 2000 ]; do
    t=$((t+1)); [ "$t" -gt 120 ] && { say "!! $(gpu_used) MiB still held"; return 1; }
    say "waiting for VRAM ($(gpu_used) MiB held)…"; sleep 5
  done; return 0
}
# A quantized artifact the size of the original is quantization that did NOT happen.
# This check is here because its absence cost two full GPU runs on the 26B.
check_shrank() {
  local dir="$1" limit_gb="$2"
  local gb; gb=$(du -sb "$dir" 2>/dev/null | cut -f1)
  gb=$(( gb / 1000000000 ))
  say "quantized size: ${gb} GB (must be < ${limit_gb} GB to be real)"
  if [ "$gb" -ge "$limit_gb" ]; then
    say "!! QUANTIZATION DID NOT SHRINK THE MODEL — same failure as the 26B MoE."
    say "!! Inspect dtypes before believing any metric from it:"
    say "!!   python -c \"import json,struct;f=open('$dir/model.safetensors','rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));import collections;c=collections.Counter(v['dtype'] for k,v in h.items() if k!='__metadata__');print(c)\""
    return 1
  fi
  return 0
}

if [ "${1:-run}" = "status" ]; then
  pgrep -f "run_smaller.sh _child" >/dev/null 2>&1 && echo "CHAIN: RUNNING" || echo "CHAIN: not running"
  echo "base: $BASE"
  echo "step: $(cat "$STATE" 2>/dev/null || echo 'not started')"
  echo "gpu:  $(gpu_used) MiB"
  echo
  for a in "$SFT:1 sft" "$ONPOLICY:2 on-policy negatives" "$DPO:3 dpo" "$GATE:4 gate (bf16)" \
           "$MEASURE:5 replies+ppl" "$MERGED:6 merged" "$QUANT:7 quantized" "$QGATE:8 gate (4bit)"; do
    p="${a%%:*}"; n="${a##*:}"
    [ -e "$p" ] && echo "  [done]    $n" || echo "  [pending] $n"
  done
  echo
  [ -f "$LOG" ] && tr '\r' '\n' < "$LOG" | grep -v 'Loading weights\|it/s\]' | tail -14 \
                || echo "(not started)"
  exit 0
fi

if [ "${1:-run}" = "stop" ]; then
  for pid in $(pgrep -f "run_smaller.sh _child|v2[.]sft_train|v2[.]dpo_train|v2[.]onpolicy|v2[.]quant_eval|v2[.]quantize|eval[.]six_gate|v2[.]serve_vllm" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue; echo "killing $pid"; kill -9 "$pid" 2>/dev/null
  done
  echo stopped; exit 0
fi

if [ "${1:-run}" != "_child" ]; then
  for pid in $(pgrep -f "v2[.]sft_train|v2[.]dpo_train|v2[.]onpolicy|v2[.]quant_eval|v2[.]quantize|eval[.]six_gate" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue; echo "killing stale $pid"; kill -9 "$pid" 2>/dev/null
  done
  sleep 3
  nohup bash "$0" _child > "$LOG" 2>&1 &
  echo "started (pid $!) -> $LOG"
  echo "base: $BASE"
  echo "check with:  bash run_smaller.sh status"
  exit 0
fi

# ---------------------------------------------------------------- the chain
say "=== smaller-base experiment: $BASE ==="
say "gpu $(gpu_used) MiB · disk $(df -h . | tail -1 | awk '{print $4}') free"
say "comparing against the banked 26B baseline: eval/six_gate_gemma_k3.json (all modes 1.000)"

# 0. Fetch the base once, up front, so a download failure costs no GPU time.
echo "0 download" > "$STATE"
say "stage 0/8 download $BASE"
python - <<PY
import os
os.environ.pop("HF_HUB_OFFLINE", None)
from huggingface_hub import snapshot_download
p = snapshot_download("$BASE", allow_patterns=["*.safetensors","*.json","*.model","*.jinja","tokenizer*"])
print("base at", p)
PY
[ $? -ne 0 ] && { say "stage 0 FAILED — download"; exit 1; }

# 1. SFT on the SAME pairs the 26B trained on.
if [ -e "$SFT" ]; then say "stage 1 sft: already done — skipping"; else
  echo "1 sft" > "$STATE"; wait_for_gpu || exit 1
  say "stage 1/8 SFT (3 epochs, same 4,999 pairs)"
  python -m v2.sft_train --data "$PAIRS" --epochs 3 --bs 4 --ga 8 --out "$SFT"
  rc=$?; [ $rc -ne 0 ] && { say "stage 1 FAILED rc=$rc — re-run to resume"; exit $rc; }
fi

# 2. On-policy negatives sampled from THIS model — the fix that made DPO real on the 26B.
#    Canned negatives taught it "boilerplate = bad" and rewards hit 1.0 by step 50.
if [ -e "$ONPOLICY" ]; then say "stage 2 on-policy: already done — skipping"; else
  echo "2 onpolicy" > "$STATE"; wait_for_gpu || exit 1
  say "stage 2/8 on-policy negatives from the SFT model"
  python -m v2.onpolicy_negatives --in "$PAIRS" --sft "$SFT" --out "$ONPOLICY" --batch 16
  rc=$?; [ $rc -ne 0 ] && { say "stage 2 FAILED rc=$rc — re-run to resume"; exit $rc; }
fi

# 3. DPO — conservative, 1 epoch; it overfits fast.
if [ -e "$DPO" ]; then say "stage 3 dpo: already done — skipping"; else
  echo "3 dpo" > "$STATE"; wait_for_gpu || exit 1
  say "stage 3/8 DPO"
  python -m v2.dpo_train --data "$ONPOLICY" --sft "$SFT" --bs 2 --ga 8 --out "$DPO"
  rc=$?; [ $rc -ne 0 ] && { say "stage 3 FAILED rc=$rc — re-run to resume"; exit $rc; }
fi

# 4. The same 99-probe gate the 26B passed. k=1 first: if it fails here, k=3 is wasted.
if [ -e "$GATE" ]; then say "stage 4 gate: already done — skipping"; else
  echo "4 gate" > "$STATE"; wait_for_gpu || exit 1
  say "stage 4/8 deploy gate, bf16, k=1"
  SATSANG_GEMMA_ADAPTER="$PWD/$DPO" python -m eval.six_gate --backend gemma --judge none \
      --k 1 --out "$GATE"
  rc=$?; [ $rc -ne 0 ] && { say "stage 4 FAILED rc=$rc — re-run to resume"; exit $rc; }
fi

# 5. Perplexity + greedy replies on the SAME held-out split as the 26B, so quality is
#    comparable without a judge (API credits are exhausted). 26B bf16 = 2.6138.
if [ -e "$MEASURE" ] && [ -e "$MEASURE_PPL" ]; then
  say "stage 5 measure: already done — skipping"; else
  echo "5 measure" > "$STATE"; wait_for_gpu || exit 1
  say "stage 5/8 greedy replies + completion-only perplexity (26B bf16 baseline = 2.6138)"
  SATSANG_GEMMA_ADAPTER="$PWD/$DPO" python -m v2.quant_eval measure --out "$MEASURE" --n 99
  rc=$?; [ $rc -ne 0 ] && { say "stage 5 FAILED rc=$rc — re-run to resume"; exit $rc; }
fi

# 6-8. Only worth doing if the model is actually good. Quantize, PROVE IT SHRANK, re-gate.
if [ -e "$MERGED" ]; then say "stage 6 merge: already done — skipping"; else
  echo "6 merge" > "$STATE"; wait_for_gpu || exit 1
  say "stage 6/8 merge adapter into the base"
  python -m v2.serve_vllm merge --adapter "$DPO" --out "$MERGED"
  rc=$?; [ $rc -ne 0 ] && { say "stage 6 FAILED rc=$rc"; rm -rf "$MERGED"; exit $rc; }
fi

if [ -e "$QUANT" ]; then say "stage 7 quantize: already done — skipping"; else
  echo "7 quantize" > "$STATE"; wait_for_gpu || exit 1
  say "stage 7/8 quantize to 4-bit (DENSE model — this should actually work)"
  python -m v2.quantize --model "$MERGED" --out "$QUANT" --method bnb --bits 4
  rc=$?; [ $rc -ne 0 ] && { say "stage 7 FAILED rc=$rc"; rm -rf "$QUANT"; exit $rc; }
  # THE CHECK THAT WAS MISSING LAST TIME.
  check_shrank "$QUANT" 15 || { say "stage 7 produced a non-quantized artifact — stopping"; exit 1; }
fi

if [ -e "$QGATE" ]; then say "stage 8 4bit gate: already done — skipping"; else
  echo "8 4bit-gate" > "$STATE"; wait_for_gpu || exit 1
  say "stage 8/8 gate the QUANTIZED model — the one that would actually ship"
  SATSANG_GEMMA_MODEL="$PWD/$QUANT" python -m eval.six_gate --backend gemma --judge none \
      --k 1 --out "$QGATE"
  rc=$?; [ $rc -ne 0 ] && { say "stage 8 FAILED rc=$rc — re-run to resume"; exit $rc; }
fi

echo done > "$STATE"
say "=== COMPLETE ==="
say "compare against the 26B:"
say "  python -m eval.watch_gates --out $GATE  --total 99 --fails      # ${TAG} bf16"
say "  python -m eval.watch_gates --out $QGATE --total 99 --fails      # ${TAG} 4-bit"
say "  python -m eval.watch_gates --out eval/six_gate_gemma_k3.json --total 297  # 26B"
say "  python -m v2.quant_eval compare --a v2/data/quant_bf16.json --b $MEASURE \\"
say "        --ppl-a v2/data/quant_bf16_ppl.json --ppl-b $MEASURE_PPL"
