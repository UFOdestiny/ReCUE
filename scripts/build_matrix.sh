#!/usr/bin/env bash
# Build the generation + probe matrix for the ACD experiments.
#
# All paths come from the repo-root .env (anonymized); nothing is hardcoded.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/build_matrix.sh scripts/matrix.txt
# where matrix.txt has one job per line:  model|dataset|max_tokens|thinking|tag
#
# Each line runs: acd.generate -> acd.probe -> scripts.run_probe_confidence,
# skipping any stage whose output already exists (idempotent / resumable).
set -u
cd "$(dirname "$0")/.."

# load .env
set -a; [ -f .env ] && . ./.env; set +a
ROOT="${EXP_ROOT:?set EXP_ROOT in .env}"

QUEUE="${1:?usage: build_matrix.sh <queue_file>}"
K="${ACD_K:-8}"
MAXLEN="${ACD_MAX_MODEL_LEN:-14336}"

while IFS='|' read -r model ds lim mt think tag; do
  [ -z "${model:-}" ] && continue
  case "$model" in \#*) continue;; esac

  if [ ! -f "$ROOT/gen/${tag}.json" ]; then
    echo "=== GEN $tag ($model $ds) ==="
    python -m acd.generate --model "$model" --dataset "$ds" --limit "$lim" --k "$K" \
      --max-tokens "$mt" --thinking "$think" --tag "$tag" \
      2>&1 | grep -aiE "^\[gen\]|Traceback|Error" | grep -v it/s
  fi

  if [ ! -f "$ROOT/probe/${tag}_probe.json" ]; then
    echo "=== PROBE $tag ==="
    python -m acd.probe --model "$model" --gen-tag "$tag" --max-probes 8 --max-model-len "$MAXLEN" \
      2>&1 | grep -aiE "saved|Traceback|Error" | grep -v it/s
  fi

  if [ ! -f "$ROOT/conf/${tag}_conf.json" ]; then
    echo "=== CONF-PROBE $tag ==="
    python -m scripts.run_probe_confidence --model "$model" --gen-tag "$tag" \
      --max-probes 8 --max-model-len "$MAXLEN" \
      2>&1 | grep -aiE "conf\]|Traceback|Error" | grep -v it/s
  fi
done < "$QUEUE"
echo "MATRIX_DONE $QUEUE"
