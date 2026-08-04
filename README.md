# TrAC: Trace-Conditioned Answer Consistency for Efficient Uncertainty Quantification in LLMs

Judge-free, single-trace uncertainty quantification (UQ) for mathematical
reasoning LLMs. TrAC reads whether a model still **re-commits** to the answer it
returned, from **one completed reasoning trace**, instead of resampling many full
trajectories.
TrAC combines two orthogonal single-trace views under a shared lightweight head:

- **PCE — Prefix-Conditioned Elicitation (active).** Re-elicit a short answer at the
  *completed* reasoning prefix (`The final answer is \boxed{...}`) and represent
  its **agreement** with the originally returned answer together with its
  **likelihood** and **confidence**. Because the reasoning prefix is unchanged,
  this reuses the vLLM KV cache and decodes only a short answer suffix — about
  **2% added latency**, not a second generation.
- **TUP — Trace Uncertainty Profile (passive).** Summarize token-level
  uncertainty already available from the primary generation (binned log-prob and
  entropy shape, slopes, extrema, low-confidence fraction). No extra decoding.

Correctness labels come from a **deterministic verifier** (`math_verify` for math,
exact option-letter match for multiple choice).

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


Or drive a single stage directly (everything is a `python -m` module):

```bash
python -m recue.generate --model Qwen3-8B --dataset math500 --k 8 --tag math500_qwen8b_k8
python -m recue.probe    --model Qwen3-8B --gen-tag math500_qwen8b_k8
python -m scripts.run_probe_confidence --model Qwen3-8B --gen-tag math500_qwen8b_k8
python -m scripts.build_cache
python -m experiments.recompute_headline
```

## License

MIT — see [`LICENSE`](LICENSE).
