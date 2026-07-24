#!/usr/bin/env bash
# Reproduce every ReCUE paper number from cached generations/probes (no GPU).
# Paths from .env; all JSON/tables written under $EXP_ROOT and printed to stdout.
#   bash scripts/run_analysis.sh
#
# Prerequisite: the generation + probe matrix already exists under $EXP_ROOT
# (see scripts/build_matrix.sh and the *_worker.sh helpers). This script is a
# pure cached recompute: it derives labels/features, dumps the shared feature
# matrices, and runs the endpoint-only ARC/TUP analyses behind the paper tables.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
ROOT="${EXP_ROOT:?set EXP_ROOT in .env}"

# 1) derive labels / sampled-answers / features from raw generations
echo "[recue] building derived caches ..."
python -m scripts.build_cache        > "$ROOT/build_cache.log" 2>&1
python -m scripts.build_cdyn_cache   > "$ROOT/build_cdyn_cache.log" 2>&1

# cells = every tag that has a confidence probe (drop seed-variant tags)
TAGS=$(ls "$ROOT"/conf/*_conf.json 2>/dev/null | xargs -n1 basename | sed 's/_conf.json//' \
       | grep -vE '_s[0-9]+_' | tr '\n' ' ')
echo "[recue] cells: $TAGS"

# 2) dump the shared feature matrices consumed by the recompute scripts
echo "[recue] dumping feature matrices (ladder_feats.npz, aime_feats.npz) ..."
python -m experiments.rebuttal_ladder_dump --tags $TAGS --out "$ROOT/ladder_feats.npz" \
    > "$ROOT/ladder_feats.log" 2>&1

# 3) headline effectiveness: RQ1 AUROC/AURC + RQ2 ablation ladder (Tables 1-3)
python -m experiments.recompute_headline    > "$ROOT/recompute_headline.log" 2>&1
python -m experiments.dump_endpoint_tables   > "$ROOT/endpoint_tables.log" 2>&1
python -m experiments.fill_aurc              > "$ROOT/fill_aurc.log" 2>&1
python -m experiments.rebuttal_aurc_table    > "$ROOT/aurc_table.log" 2>&1

# 4) RQ2 mechanism controls: ARC vs P(True), teacher-forced support, decoding matrix
python -m experiments.rebuttal_arc_vs_ptrue  > "$ROOT/arc_vs_ptrue.log" 2>&1
python -m experiments.analyze_teacherforce   > "$ROOT/teacherforce.log" 2>&1
python -m experiments.analyze_decmatrix      > "$ROOT/decmatrix.log" 2>&1

# 5) RQ3 consensus coverage + RQ4 transfer / generality
python -m experiments.novelty_amplifiers --tags $TAGS --exp all > "$ROOT/amplifiers.log" 2>&1
python -m experiments.transfer_recue         > "$ROOT/transfer_recue.log" 2>&1
python -m experiments.rebuttal_nogpu         > "$ROOT/rebuttal_nogpu.log" 2>&1
python -m experiments.aime_analysis          > "$ROOT/aime_analysis.log" 2>&1

# 6) robustness / diagnostics (Appendix)
python -m experiments.variance               > "$ROOT/variance.log" 2>&1
python -m experiments.cue_ensemble           > "$ROOT/cue_ensemble.log" 2>&1
python -m experiments.confound               > "$ROOT/confound.log" 2>&1
python -m experiments.recompute_measurement  > "$ROOT/measurement.log" 2>&1
python -m experiments.recompute_capacity     > "$ROOT/capacity.log" 2>&1
python -m experiments.non_math_generalization > "$ROOT/nonmath.log" 2>&1

# 7) efficiency (Table 7 / Appendix E) -- rebuttal_latency_stats reads cached timings
python -m experiments.rebuttal_latency_stats > "$ROOT/latency_stats.log" 2>&1

echo "RECUE_ANALYSIS_DONE"
