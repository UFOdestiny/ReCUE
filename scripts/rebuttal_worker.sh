#!/usr/bin/env bash
# Run rebuttal probes for a list of tags on ONE gpu. Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal_worker.sh <queuefile>
set -u
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
ROOT="${EXP_ROOT:?}"; mkdir -p "$ROOT/rebuttal"
Q="${1:?queue file}"
declare -A M=( [qwen4b]=Qwen3-4B [qwen8b]=Qwen3-8B [qwen14b]=Qwen3-14B \
  [qwen35_9b]=Qwen3.5-9B [phi4r]=Phi-4-reasoning [ministral]=Ministral-3-14B-Reasoning-2512 \
  [llama8b]=Llama-3.1-8B-Instruct )
while read -r tag; do
  [ -z "${tag:-}" ] && continue
  case "$tag" in \#*) continue;; esac
  base="${tag%_k8}"                     # strip trailing _k8
  key="${base#*_}"                      # strip dataset prefix -> modelkey
  model="${M[$key]:-}"
  if [ -z "$model" ]; then echo "SKIP $tag (no model for key=$key)"; continue; fi
  echo "=== REB $tag ($model) ==="
  python -m scripts.run_probe_rebuttal --model "$model" --gen-tag "$tag" \
     2>&1 | grep -aiE "reb\]|Traceback|Error|CUDA" | grep -v it/s
done < "$Q"
echo "REB_DONE $Q"
