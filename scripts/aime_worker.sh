#!/usr/bin/env bash
# Full AIME pipeline for a list of model keys on ONE gpu. Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/aime_worker.sh qwen4b qwen8b qwen14b
# Builds, per model, tag=aime_<key>_k8:
#   gen -> probe -> conf-probe -> labels/sampans -> cdyn -> ptrue -> rebuttal
# Idempotent: every stage skips if its output already exists.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
ROOT="${EXP_ROOT:?}"
mkdir -p "$ROOT"/{gen,probe,conf,labels,sampans,cdyn,ptrue,rebuttal}
K=8; MAXLEN="${RECUE_MAX_MODEL_LEN:-14336}"; MT=8192; DS=aime

declare -A M=( [qwen4b]=Qwen3-4B [qwen8b]=Qwen3-8B [qwen14b]=Qwen3-14B \
  [qwen35_9b]=Qwen3.5-9B [phi4r]=Phi-4-reasoning [ministral]=Ministral-3-14B-Reasoning-2512 )

for key in "$@"; do
  model="${M[$key]:-}"; [ -z "$model" ] && { echo "SKIP $key (no model)"; continue; }
  tag="aime_${key}_k8"
  echo "===== $tag ($model) ====="

  if [ ! -f "$ROOT/gen/${tag}.json" ]; then
    echo "[GEN] $tag"
    python -m recue.generate --model "$model" --dataset "$DS" --limit 0 --k "$K" \
      --max-tokens "$MT" --thinking 1 --tag "$tag" 2>&1 | grep -aiE "^\[gen\]|Traceback|Error|CUDA" | grep -v it/s
  fi
  if [ ! -f "$ROOT/probe/${tag}_probe.json" ]; then
    echo "[PROBE] $tag"
    python -m recue.probe --model "$model" --gen-tag "$tag" --max-probes 8 --max-model-len "$MAXLEN" \
      2>&1 | grep -aiE "saved|Traceback|Error|CUDA" | grep -v it/s
  fi
  if [ ! -f "$ROOT/conf/${tag}_conf.json" ]; then
    echo "[CONF] $tag"
    python -m scripts.run_probe_confidence --model "$model" --gen-tag "$tag" \
      --max-probes 8 --max-model-len "$MAXLEN" 2>&1 | grep -aiE "conf\]|Traceback|Error|CUDA" | grep -v it/s
  fi
  if [ ! -f "$ROOT/ptrue/${tag}_ptrue.json" ]; then
    echo "[PTRUE] $tag"
    python -m scripts.run_ptrue --model "$model" --gen-tag "$tag" --max-model-len "$MAXLEN" \
      2>&1 | grep -aiE "ptrue|saved|Traceback|Error|CUDA" | grep -v it/s
  fi
  if [ ! -f "$ROOT/rebuttal/${tag}_reb.json" ]; then
    echo "[REB] $tag"
    python -m scripts.run_probe_rebuttal --model "$model" --gen-tag "$tag" \
      2>&1 | grep -aiE "reb\]|Traceback|Error|CUDA" | grep -v it/s
  fi
done
echo "AIME_WORKER_DONE $*"
