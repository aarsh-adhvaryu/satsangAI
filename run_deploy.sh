#!/usr/bin/env bash
# SatsangAI V2 — the whole deploy chain, one command, detached and resumable.
#
#   bash run_deploy.sh            # start (or resume) the chain
#   bash run_deploy.sh status     # where is it
#   bash run_deploy.sh stop       # kill it
#
# Exists because the individual commands kept failing for reasons that had nothing to do
# with the model:
#   * multi-line pastes were garbling in this terminal (doubled/dropped characters, an
#     unterminated quote leaving zsh at `dquote>`), so the chain is a FILE, not a paste;
#   * the k=3 run was launched in the foreground through `tee` and died silently with no
#     traceback — the signature of a SIGKILL/SIGHUP, not a crash — so every step now runs
#     under nohup, detached from the terminal;
#   * one 52 GB model fits on this card, so the steps must be SEQUENTIAL and each must
#     wait for the previous to release VRAM. Two at once wedges the loader for ten
#     minutes and then dies.
#
# Every step is skipped if its output already exists, so re-running after any failure
# continues rather than restarting. The gate steps additionally resume per-reply from
# their own sidecar.
set -uo pipefail
cd "$(dirname "$0")"

K3_OUT=eval/six_gate_gemma_k3.json          # step 1 — reproducible bf16 number
MERGED=v2/data/gemma4-v2-merged             # step 2 — adapter baked into the base
QUANT=v2/data/gemma4-v2-4bit                # step 3 — 4-bit for commodity GPUs
Q_OUT=eval/six_gate_gemma_4bit.json         # step 4 — does 4-bit hold the gates
LOG=eval/deploy_chain.log
STATE=eval/.deploy_chain_step

say() { echo "[$(date +%H:%M:%S)] $*"; }

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0; }

wait_for_gpu() {          # the preflight whose absence cost a whole run today
  local tries=0
  while [ "$(gpu_used)" -gt 2000 ]; do
    tries=$((tries + 1))
    if [ "$tries" -gt 60 ]; then
      say "!! $(gpu_used) MiB still held after 5 min. Holder:"
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv
      return 1
    fi
    say "waiting for VRAM to free ($(gpu_used) MiB held)…"
    sleep 5
  done
  return 0
}

# ---------------------------------------------------------------- status / stop
if [ "${1:-run}" = "status" ]; then
  if pgrep -f "run_deploy.sh run|deploy_chain" >/dev/null 2>&1; then
    echo "CHAIN: RUNNING"
  else
    echo "CHAIN: not running"
  fi
  echo "step:  $(cat "$STATE" 2>/dev/null || echo 'not started')"
  echo "gpu:   $(gpu_used) MiB"
  echo "disk:  $(df -h . | tail -1 | awk '{print $4" free"}')"
  echo
  for a in "$K3_OUT:1 k3-gate" "$MERGED:2 merged" "$QUANT:3 quantized" "$Q_OUT:4 4bit-gate"; do
    p="${a%%:*}"; n="${a##*:}"
    [ -e "$p" ] && echo "  [done]    $n  ($p)" || echo "  [pending] $n  ($p)"
  done
  echo
  if [ -f "$LOG" ]; then
    echo "--- last log lines ---"
    tr '\r' '\n' < "$LOG" | grep -v 'Loading weights\|it/s\]' | tail -8
    echo "--- load progress ---"
    tr '\r' '\n' < "$LOG" | grep -o 'Loading weights: *[0-9]*%[^]]*]' | tail -1
  else
    echo "(no log yet — chain has not started)"
  fi
  echo
  python -m eval.watch_gates --out "$K3_OUT" --total 297 2>/dev/null | tail -12
  exit 0
fi

if [ "${1:-run}" = "stop" ]; then
  for pid in $(pgrep -f "eval[.]six_gate|v2[.]serve_vllm|v2[.]quantize|run_deploy" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    echo "killing $pid"; kill -9 "$pid" 2>/dev/null
  done
  echo "stopped."
  exit 0
fi

# ---------------------------------------------------------------- detach
# Re-exec ourselves in the background exactly once, so the chain outlives the terminal.
if [ "${1:-run}" != "_child" ]; then
  for pid in $(pgrep -f "eval[.]six_gate|v2[.]quantize" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    echo "killing stale $pid"; kill -9 "$pid" 2>/dev/null
  done
  sleep 3
  nohup bash "$0" _child > "$LOG" 2>&1 &
  echo "chain started (pid $!) -> $LOG"
  echo
  echo "check on it with:   bash run_deploy.sh status"
  exit 0
fi

# ---------------------------------------------------------------- the chain
say "=== SatsangAI V2 deploy chain ==="
say "gpu $(gpu_used) MiB | disk $(df -h . | tail -1 | awk '{print $4}') free"

# STEP 1 — reproducible bf16 gate (k=3). Resumes per-reply from its sidecar.
if [ -e "$K3_OUT" ]; then
  say "step 1 k3-gate: already done ($K3_OUT) — skipping"
else
  echo "1 k3-gate" > "$STATE"
  wait_for_gpu || exit 1
  say "step 1/4 k3-gate: 99 probes x 3 draws, deterministic gates, no API"
  python -m eval.six_gate --backend gemma --judge none --k 3 --out "$K3_OUT"
  rc=$?
  [ $rc -ne 0 ] && { say "step 1 FAILED (rc=$rc) — re-run this script to resume"; exit $rc; }
  say "step 1 done"
fi

# STEP 2 — merge the LoRA into the base so there is one standalone artifact.
if [ -f "$MERGED/config.json" ] && ls "$MERGED"/*.safetensors >/dev/null 2>&1; then
  say "step 2 merge: already done ($MERGED) — skipping"
else
  echo "2 merge" > "$STATE"
  wait_for_gpu || exit 1
  say "step 2/4 merge: baking gemma4-v2-dpo2-lora into the base (~52 GB written)"
  python -m v2.serve_vllm merge --adapter v2/data/gemma4-v2-dpo2-lora --out "$MERGED"
  rc=$?
  [ $rc -ne 0 ] && { say "step 2 FAILED (rc=$rc)"; rm -rf "$MERGED"; exit $rc; }
  say "step 2 done"
fi

# STEP 3 — quantize to 4-bit. `bnb` not `awq`: awq and vllm are NOT installed in this
# environment, and bnb answers the only question that matters first — does quality hold.
if [ -f "$QUANT/config.json" ] && ls "$QUANT"/*.safetensors >/dev/null 2>&1; then
  say "step 3 quantize: already done ($QUANT) — skipping"
else
  echo "3 quantize" > "$STATE"
  wait_for_gpu || exit 1
  say "step 3/4 quantize: 4-bit nf4 via bitsandbytes (~13 GB out)"
  python -m v2.quantize --model "$MERGED" --out "$QUANT" --method bnb
  rc=$?
  [ $rc -ne 0 ] && { say "step 3 FAILED (rc=$rc)"; rm -rf "$QUANT"; exit $rc; }
  say "step 3 done"
fi

# STEP 4 — the step that did not exist until today: prove the quantized model still
# passes. SATSANG_GEMMA_MODEL points the serving path at standalone weights instead of
# base+adapter; without it nothing could load these weights and "prove 4-bit" was a
# sentence in a docstring with no code behind it.
if [ -e "$Q_OUT" ]; then
  say "step 4 4bit-gate: already done ($Q_OUT) — skipping"
else
  echo "4 4bit-gate" > "$STATE"
  wait_for_gpu || exit 1
  say "step 4/4 4bit-gate: same 99 probes against the QUANTIZED model"
  SATSANG_GEMMA_MODEL="$PWD/$QUANT" \
    python -m eval.six_gate --backend gemma --judge none --k 1 --out "$Q_OUT"
  rc=$?
  [ $rc -ne 0 ] && { say "step 4 FAILED (rc=$rc) — re-run to resume"; exit $rc; }
  say "step 4 done"
fi

echo "done" > "$STATE"
say "=== CHAIN COMPLETE ==="
say "bf16  gate: $K3_OUT"
say "4-bit gate: $Q_OUT"
say "compare them before shipping 4-bit:"
say "  python -m eval.watch_gates --out $K3_OUT --total 297"
say "  python -m eval.watch_gates --out $Q_OUT  --total 99"
