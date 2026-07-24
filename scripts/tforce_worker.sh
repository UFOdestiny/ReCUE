#!/usr/bin/env bash
# Teacher-force original-answer probe for a queue of tags on ONE gpu.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
ROOT="${EXP_ROOT:?}"; mkdir -p "$ROOT/tforce"
Q="${1:?queue file}"
declare -A M=( [qwen4b]=Qwen3-4B [qwen8b]=Qwen3-8B [qwen14b]=Qwen3-14B \
  [qwen35_9b]=Qwen3.5-9B [phi4r]=Phi-4-reasoning [ministral]=Ministral-3-14B-Reasoning-2512 )
while read -r tag; do
  [ -z "${tag:-}" ] && continue
  case "$tag" in \#*) continue;; esac
  base="${tag%_k8}"; key="${base#*_}"; model="${M[$key]:-}"
  if [ -z "$model" ]; then echo "SKIP $tag (no model key=$key)"; continue; fi
  echo "=== TF $tag ($model) ==="
  python -m scripts.run_probe_teacherforce --model "$model" --gen-tag "$tag" \
     2>&1 | grep -aiE "tf\]|Traceback|Error|CUDA out" | grep -v it/s
done < "$Q"
echo "TF_DONE $Q"
