# Answer-Confidence Dynamics (ACD)

Judge-free, verifier-supervised uncertainty quantification (UQ) for **reasoning LLMs**.

ACD reads uncertainty from **how a model's answer evolves along a single reasoning
trace**, rather than from repeated sampling. We take one completed chain-of-thought,
cut it at reasoning-step boundaries, and cheaply force-decode
`The final answer is \boxed{...}` at each cut (reusing the vLLM prefix cache, so all
probes together cost ≈2% extra generated tokens). This yields, per step, an
intermediate **answer** and the model's **forced-answer confidence** (logprob). A
lightweight head over the resulting trajectory features predicts response correctness.

Correctness labels come from a deterministic checker (`math_verify`) — **no judge
model anywhere**, so there is no judge-induced label leakage.

## Key results

Full numbers and interpretation: [`docs/RESULTS.md`](docs/RESULTS.md).

- **Best single-generation UQ.** Macro AUROC 0.79 over 31 model×dataset cells (7 model
  families), beating every same-cost baseline (P(True), self-certainty, DeepConf,
  logprob) and the answer-convergence prior-art by +0.15.
- **The signal is trajectory dynamics, not the endpoint.** Ablation: adding the
  confidence *trajectory* on top of answer-convergence + final-answer confidence lifts
  AUROC by +0.05 (significant in 14/31 cells); removing the final value entirely barely
  changes it.
- **Beats self-consistency at matched cost.** Fusing ACD with 8-sample self-consistency
  beats SC@8 alone (+0.02 AUROC, significant in 10/25 cells) — because ACD detects the
  *confident-consensus errors* self-consistency is structurally blind to (on
  high-agreement responses SC is at chance ~0.5, ACD stays ~0.66).

## Layout

```text
acd/                    core library
  env.py                .env loading, paths, math answer parsing & verification
  data.py               verifiable-answer datasets (GSM8K, MATH500, Minerva, Olympiad, AMC23)
  generate.py           vLLM generation of reasoning traces (+ per-sample logprobs)   [CLI]
  probe.py              intermediate-answer probe (identity trajectory)               [CLI]
  features.py           answer-stabilization / confidence-dynamics features
  baselines.py          single-pass UQ baselines (logprob, entropy, DeepConf, self-certainty)
  metrics.py            AUROC / AURC / ECE / risk-at-coverage
scripts/                runnable entrypoints & orchestration
  run_probe_confidence.py   confidence-dynamics probe (answer + forced-answer logprob) [CLI]
  run_ptrue.py              P(True) self-eval baseline                                 [CLI]
  build_cache.py            derive labels / sampled-answers / features caches
  build_matrix.sh           generate+probe a model×dataset matrix (reads .env)
  run_analysis.sh           run all paper experiments over cached cells
  matrix.txt                example job list for build_matrix.sh
experiments/            main-paper analyses (main comparison, ablations, mechanism, variance, hybrid)
analysis/               exploratory studies & negative results (extra probes, alternative baselines)
docs/RESULTS.md         consolidated results & interpretation
```

## Setup

```bash
pip install -r requirements.txt          # torch, vllm, transformers, scikit-learn, math_verify, ...
cp .env.example .env                      # then edit paths (see below)
```

All paths are read from a gitignored `.env` (anonymized; no absolute user paths in code):

```ini
MODELS_ROOT=/path/to/models          # dir of local model folders (names match --model)
DATASETS_ROOT=/path/to/hf_hub_cache  # HuggingFace datasets cache
EXP_ROOT=/path/to/experiment_outputs # all caches & results land here
```

## Quick start

```bash
# 1. generate reasoning traces + intermediate-answer/confidence probes for a matrix
CUDA_VISIBLE_DEVICES=0 bash scripts/build_matrix.sh scripts/matrix.txt

# 2. run all analyses over whatever cells exist in $EXP_ROOT
bash scripts/run_analysis.sh
```

Or drive a single stage directly (everything is a `python -m` module):

```bash
python -m acd.generate --model Qwen3-8B --dataset math500 --k 8 --tag math500_qwen8b_k8
python -m acd.probe                 --model Qwen3-8B --gen-tag math500_qwen8b_k8
python -m scripts.run_probe_confidence --model Qwen3-8B --gen-tag math500_qwen8b_k8
python -m scripts.build_cache
python -m experiments.main_comparison --tags math500_qwen8b_k8 --out $EXP_ROOT/main.json
```

## Caches (under `$EXP_ROOT`)

| dir | contents |
|-----|----------|
| `gen/`      | reasoning traces + k self-consistency samples + logprobs |
| `probe/`    | intermediate-answer trajectory (identity) |
| `conf/`     | intermediate-answer + forced-answer confidence trajectory |
| `labels/`   | deterministic correctness labels (`math_verify`) |
| `sampans/`  | extracted answers of the k samples (for self-consistency) |
| `feats/`, `cdyn/` | precomputed convergence / confidence-dynamics features |
| `ptrue/`    | P(True) baseline scores |

Once caches exist, all `experiments/` and `analysis/` scripts recompute in seconds
without any GPU.
