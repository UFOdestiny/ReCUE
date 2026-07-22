#!/usr/bin/env bash
# Run the full analysis pipeline over whatever cells exist in EXP_ROOT.
# Paths from .env; results/logs written under $EXP_ROOT.
#   bash scripts/run_analysis.sh
set -u
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
ROOT="${EXP_ROOT:?set EXP_ROOT in .env}"

# 1) build derived caches (labels, sampled answers, features, confidence features)
echo "[analysis] building caches ..."
python -m scripts.build_cache          > "$ROOT/build_cache.log" 2>&1

# tags = every cell that has a confidence probe
TAGS=$(ls "$ROOT"/conf/*_conf.json 2>/dev/null | xargs -n1 basename | sed 's/_conf.json//' \
       | grep -vE '_s[0-9]+_' | tr '\n' ' ')
echo "[analysis] cells: $TAGS"

# 2) main single-generation SOTA comparison
python -m experiments.main_comparison   --tags $TAGS --out "$ROOT/main_comparison.json" \
    > "$ROOT/main_comparison.log" 2>&1

# 3) decisive trajectory ablation (dynamics vs endpoint)
python -m experiments.ablation_trajectory --tags $TAGS --out "$ROOT/ablation_trajectory.json" \
    > "$ROOT/ablation_trajectory.log" 2>&1

# 4) mechanism: confidence trajectories + high-agreement blind-spot
python -m experiments.mechanism_plots    --tags $TAGS > "$ROOT/mechanism_plots.log" 2>&1
python -m experiments.mechanism_highconf              > "$ROOT/mechanism_highconf.log" 2>&1

# 5) matched-budget beat-SOTA fusion
python -m experiments.hybrid_sota        --tags $TAGS > "$ROOT/hybrid_sota.log" 2>&1

echo "MATRIX_ANALYSIS_DONE"
