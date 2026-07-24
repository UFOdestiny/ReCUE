#!/usr/bin/env bash
# Run decoding-matrix conditions for one (model,dataset,tag) on ONE gpu.
# Queue lines: model|dataset|tag|primary_temp|probe_temp
set -u
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
Q="${1:?queue file}"
while IFS='|' read -r model dataset tag ptemp qtemp; do
  [ -z "${model:-}" ] && continue
  case "$model" in \#*) continue;; esac
  echo "=== DM $tag p=$ptemp q=$qtemp ($model/$dataset) ==="
  python -m scripts.run_decoding_matrix --model "$model" --dataset "$dataset" \
     --tag "$tag" --primary-temp "$ptemp" --probe-temp "$qtemp" --limit 300 \
     2>&1 | grep -aiE "dm\]|Traceback|Error|CUDA out" | grep -v it/s
done < "$Q"
echo "DM_DONE $Q"
