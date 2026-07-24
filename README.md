# ReCUE: Answer Re-Commitment for Compute-Aware Uncertainty in Reasoning Models

Judge-free, single-trace uncertainty quantification (UQ) for mathematical
reasoning LLMs. ReCUE reads whether a model still **re-commits** to the answer it
returned, from **one completed reasoning trace**, instead of resampling many full
trajectories.

ReCUE combines two orthogonal single-trace views under a shared lightweight head:

- **ARC — Answer Re-Commitment (active).** Re-elicit a short answer at the
  *completed* reasoning prefix (`The final answer is \boxed{...}`) and represent
  its **agreement** with the originally returned answer together with its
  **likelihood** and **confidence**. Because the reasoning prefix is unchanged,
  this reuses the vLLM KV cache and decodes only a short answer suffix — about
  **3% added latency**, not a second generation.
- **TUP — Trace Uncertainty Profile (passive).** Summarize token-level
  uncertainty already available from the primary generation (binned log-prob and
  entropy shape, slopes, extrema, low-confidence fraction). No extra decoding.

Correctness labels come from a **deterministic verifier** (`math_verify` for math,
exact option-letter match for multiple choice) — **no LLM judge anywhere**, so
there is no judge-induced label leakage.

## Key results

Across **30 model–dataset cells** (six reasoning models × five math benchmarks,
eight samples per problem):

- **Matches 8-sample self-consistency at ~1× cost.** ReCUE reaches **0.894 macro
  AUROC** from one completed trace plus one cached answer probe, statistically
  matching SC@8 (0.878) while adding only **3% measured latency**.
- **Best single-trace estimator.** ReCUE improves on every single-trace baseline
  (mean log-prob, self-certainty, DeepConf, P(True), and the passive TUP), and the
  active ARC module alone already ranks above all prior single-trace estimators.
- **Covers the consensus blind spot.** When sampled answers are unanimous, vote
  fraction is uninformative by construction (AUROC 0.500) yet ~3% of unanimous
  answers are still wrong. Adding ReCUE to SC@8 (**ConsensusFusion**) raises macro
  AUROC from 0.878 to **0.916** at a matched eight-sample budget.
- **Distinct from self-verification.** A single re-commitment agreement bit is
  competitive with P(True); the full ARC view significantly surpasses it (+0.051),
  and re-generating an answer carries information beyond teacher-forcing it.

## Layout

```text
recue/                    core library
  env.py                  .env loading, paths, math answer parsing & verification
  data.py                 verifiable math datasets (GSM8K, MATH500, Minerva, Olympiad, AIME, AMC23)
  data_nonmath.py         multiple-choice reasoning tasks (BBH, GPQA) for generality checks
  generate.py             vLLM generation of primary trace + k samples (+ logprobs)   [CLI]
  probe.py                re-elicitation probe at reasoning-prefix cuts               [CLI]
  features.py             ARC / TUP feature construction from a single trace
  baselines.py            single-trace UQ baselines (logprob, entropy, DeepConf, self-certainty)
  metrics.py              AUROC / AURC / ECE / risk-at-coverage
scripts/                  runnable entrypoints & orchestration
  build_matrix.sh         generate -> probe -> confidence probe over a model×dataset queue
  run_probe_confidence.py per-cut answer + forced-answer first-token logprob          [CLI]
  run_probe_rebuttal.py   full-answer greedy decode with logprobs (ARC likelihood)    [CLI]
  run_probe_teacherforce.py teacher-forced original-answer support (RQ2 control)      [CLI]
  run_probe_multicue.py   3-cue robustness probe                                      [CLI]
  run_decoding_matrix.py  primary×probe temperature matrix (RQ2)                      [CLI]
  run_ptrue.py            P(True) self-evaluation baseline                            [CLI]
  build_cache.py          derive labels / sampled-answers / feature caches
  build_cdyn_cache.py     derive confidence-dynamics feature cache
  run_analysis.sh         reproduce every paper number from cached generations (no GPU)
  *_worker.sh             single-GPU queue runners; matrix*.txt are their job lists
experiments/              cached analyses that produce the paper tables & figures
visualization/            headline figures (Pareto, risk-coverage)
```

## Setup

```bash
pip install -r requirements.txt          # torch, vllm, transformers, scikit-learn, math_verify, ...
cp .env.example .env                      # then edit the paths below
```

All paths are read from a gitignored `.env` (anonymized; no absolute user paths
in code):

```ini
MODELS_ROOT=/path/to/models              # dir of local model folders (names match --model)
DATASETS_ROOT=/path/to/hf_datasets_hub   # HuggingFace datasets cache
EXP_ROOT=/path/to/experiment_outputs     # all caches, results, and logs land here
```

## Quick start

```bash
# 1. build the generation + probe matrix (GPU); idempotent / resumable
CUDA_VISIBLE_DEVICES=0 bash scripts/build_matrix.sh scripts/matrix.txt

# 2. add the ARC likelihood, P(True), and teacher-forced probes for each cell
CUDA_VISIBLE_DEVICES=0 bash scripts/rebuttal_worker.sh scripts/matrix.txt
CUDA_VISIBLE_DEVICES=0 bash scripts/tforce_worker.sh  scripts/matrix.txt

# 3. reproduce every paper number from the caches (no GPU)
bash scripts/run_analysis.sh
```

Or drive a single stage directly (everything is a `python -m` module):

```bash
python -m recue.generate --model Qwen3-8B --dataset math500 --k 8 --tag math500_qwen8b_k8
python -m recue.probe    --model Qwen3-8B --gen-tag math500_qwen8b_k8
python -m scripts.run_probe_confidence --model Qwen3-8B --gen-tag math500_qwen8b_k8
python -m scripts.build_cache
python -m experiments.recompute_headline
```

## Caches (under `$EXP_ROOT`)

| dir / file | contents |
|-----|----------|
| `gen/`       | primary reasoning trace + k self-consistency samples + logprobs |
| `probe/`     | re-elicited answers at reasoning-prefix cuts |
| `conf/`      | per-cut answer + forced-answer first-token logprob |
| `rebuttal/`  | full-answer greedy decode with token logprobs (ARC likelihood) |
| `tforce/`    | teacher-forced original-answer support (RQ2 control) |
| `ptrue/`     | P(True) self-evaluation baseline scores |
| `decmatrix/` | primary×probe temperature-matrix regenerations (RQ2) |
| `labels/`    | deterministic correctness labels (`math_verify` / exact-match) |
| `sampans/`   | extracted answers of the k samples (for self-consistency) |
| `feats/`, `cdyn/` | precomputed ARC / TUP feature caches |
| `ladder_feats.npz`, `aime_feats.npz` | shared per-cell feature matrices consumed by the recompute scripts |

Once the caches exist, all `experiments/` scripts and `scripts/run_analysis.sh`
recompute in seconds without a GPU.

## Citation

Paper title: *ReCUE: Answer Re-Commitment for Compute-Aware Uncertainty in
Mathematical Reasoning Models*. (Anonymous submission; citation to be added.)

## License

MIT — see [`LICENSE`](LICENSE).
