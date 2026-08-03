#!/usr/bin/env bash
# Quantization before/after — the full measurement, unattended.
#
#   bash run_quant.sh          # start (or resume)
#   bash run_quant.sh status   # where is it
#   bash run_quant.sh stop     # kill it
#
# Runs detached under nohup, so closing the laptop does not touch it. Every stage writes
# its result to disk and is SKIPPED on re-run if that file exists, so if anything dies you
# re-run this same command and it continues. The reply stages additionally resume
# per-probe from their own output file.
#
# WHY THIS EXISTS RATHER THAN "re-run the gates before and after": every deterministic gate
# is at 1.000 across 297 draws. A saturated metric cannot detect a regression — 4-bit could
# lose real warmth and depth and still score 1.000 on "invents no citation". The gates run
# here only as a FLOOR (did anything gross break); perplexity and greedy divergence are the
# instruments that actually have headroom. See v2/quant_eval.py.
#
# ONE MODEL AT A TIME: 52 GB bf16 or ~13 GB 4-bit, never concurrently, so stages are
# strictly sequential and each waits for VRAM to drain before starting.
set -uo pipefail
cd "$(dirname "$0")"

# BITS=8 bash run_quant.sh  -> the same chain at 8-bit, into its own set of files, so the
# 4-bit results are never overwritten and the two can be compared side by side.
# Measured 2026-08-03: 4-bit costs +6.92% perplexity, over the 5% line, so 8-bit is the
# next thing to try before giving up on commodity hardware.
BITS="${BITS:-4}"
MERGED=v2/data/gemma4-v2-merged
QUANT=v2/data/gemma4-v2-${BITS}bit
BF16_REP=v2/data/quant_bf16.json
BF16_PPL=v2/data/quant_bf16_ppl.json
Q_REP=v2/data/quant_${BITS}bit.json
Q_PPL=v2/data/quant_${BITS}bit_ppl.json
Q_GATE=eval/six_gate_gemma_${BITS}bit.json
REPORT=v2/data/quant_report_${BITS}bit.json
LOG=eval/quant_chain_${BITS}bit.log
STATE=eval/.quant_chain_step_${BITS}bit

say() { echo "[$(date +%F' '%H:%M:%S)] $*"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0; }

wait_for_gpu() {
  local t=0
  while [ "$(gpu_used)" -gt 2000 ]; do
    t=$((t+1)); [ "$t" -gt 60 ] && { say "!! $(gpu_used) MiB still held"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; return 1; }
    say "waiting for VRAM ($(gpu_used) MiB held)…"; sleep 5
  done
  return 0
}

if [ "${1:-run}" = "status" ]; then
  pgrep -f "run_quant.sh _child" >/dev/null 2>&1 && echo "CHAIN: RUNNING" || echo "CHAIN: not running"
  echo "step: $(cat "$STATE" 2>/dev/null || echo 'not started')"
  echo "gpu:  $(gpu_used) MiB"
  echo
  for a in "$QUANT:1 quantized" "$BF16_REP:2 bf16 replies" "$BF16_PPL:2 bf16 ppl" \
           "$Q_REP:3 4bit replies" "$Q_PPL:3 4bit ppl" "$Q_GATE:4 4bit gates" "$REPORT:5 report"; do
    p="${a%%:*}"; n="${a##*:}"
    [ -e "$p" ] && echo "  [done]    $n" || echo "  [pending] $n"
  done
  echo
  if [ -f "$LOG" ]; then
    tr '\r' '\n' < "$LOG" | grep -v 'Loading weights\|it/s\]' | tail -12
  else
    echo "(not started)"
  fi
  exit 0
fi

if [ "${1:-run}" = "stop" ]; then
  for pid in $(pgrep -f "v2[.]quant_eval|eval[.]six_gate|v2[.]quantize|run_quant.sh _child" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue; echo "killing $pid"; kill -9 "$pid" 2>/dev/null
  done
  echo stopped; exit 0
fi

if [ "${1:-run}" != "_child" ]; then
  for pid in $(pgrep -f "v2[.]quant_eval|eval[.]six_gate|v2[.]quantize" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue; echo "killing stale $pid"; kill -9 "$pid" 2>/dev/null
  done
  sleep 3
  nohup bash "$0" _child > "$LOG" 2>&1 &
  echo "started (pid $!) -> $LOG"
  echo "check with:  bash run_quant.sh status"
  exit 0
fi

# ------------------------------------------------------------------ the chain
say "=== quantization before/after (${BITS}-bit) ==="
say "gpu $(gpu_used) MiB · disk $(df -h . | tail -1 | awk '{print $4}') free"

# 1. BF16 BASELINE FIRST. Deliberately before quantizing: if the chain dies later, the
#    baseline is already banked, and the baseline is the thing everything is measured
#    against. Uses base+adapter (no --model), which is what is serving today.
if [ -e "$BF16_REP" ] && [ -e "$BF16_PPL" ]; then
  say "stage 1 bf16 baseline: already done — skipping"
else
  echo "1 bf16-baseline" > "$STATE"; wait_for_gpu || exit 1
  say "stage 1/5 bf16 baseline: 99 greedy replies + perplexity on 99 held-out pairs"
  python -m v2.quant_eval measure --out "$BF16_REP" --n 99
  rc=$?; [ $rc -ne 0 ] && { say "stage 1 FAILED rc=$rc — re-run to resume"; exit $rc; }
fi

# 2. QUANTIZE. bnb/nf4: awq and vllm are not installed here, and bnb answers the only
#    question that matters first — does quality survive at all.
if [ -e "$QUANT" ]; then
  say "stage 2 quantize: already done — skipping"
else
  echo "2 quantize" > "$STATE"; wait_for_gpu || exit 1
  say "stage 2/5 quantize: ${BITS}-bit from $MERGED"
  python -m v2.quantize --model "$MERGED" --out "$QUANT" --method bnb --bits "$BITS"
  rc=$?; [ $rc -ne 0 ] && { say "stage 2 FAILED rc=$rc"; rm -rf "$QUANT"; exit $rc; }
fi

# 3. THE SAME MEASUREMENTS ON 4-BIT. Same probes, same temperature 0, same held-out pairs.
if [ -e "$Q_REP" ] && [ -e "$Q_PPL" ]; then
  say "stage 3 4bit measurement: already done — skipping"
else
  echo "3 4bit-measure" > "$STATE"; wait_for_gpu || exit 1
  say "stage 3/5 ${BITS}bit: 99 greedy replies + perplexity"
  python -m v2.quant_eval measure --model "$QUANT" --out "$Q_REP" --n 99
  rc=$?; [ $rc -ne 0 ] && { say "stage 3 FAILED rc=$rc — re-run to resume"; exit $rc; }
fi

# 4. GATES AS A FLOOR on the quantized model. Not the instrument for subtle damage, but
#    it must not have broken a hard guarantee.
if [ -e "$Q_GATE" ]; then
  say "stage 4 4bit gates: already done — skipping"
else
  echo "4 4bit-gates" > "$STATE"; wait_for_gpu || exit 1
  say "stage 4/5 ${BITS}bit gates: 99 probes, deterministic only"
  SATSANG_GEMMA_MODEL="$PWD/$QUANT" python -m eval.six_gate --backend gemma --judge none \
      --k 1 --out "$Q_GATE"
  rc=$?; [ $rc -ne 0 ] && { say "stage 4 FAILED rc=$rc — re-run to resume"; exit $rc; }
fi

# 5. COMPARE — CPU only, no model, no API.
echo "5 compare" > "$STATE"
say "stage 5/5 compare"
python -m v2.quant_eval compare --a "$BF16_REP" --b "$Q_REP" \
    --ppl-a "$BF16_PPL" --ppl-b "$Q_PPL" --out "$REPORT"

echo done > "$STATE"
say "=== COMPLETE ==="
say "report: $REPORT"
say "read it any time with:"
say "  python -m v2.quant_eval compare --a $BF16_REP --b $Q_REP --ppl-a $BF16_PPL --ppl-b $Q_PPL"
say "  python -m eval.watch_gates --out $Q_GATE --total 99 --fails"
