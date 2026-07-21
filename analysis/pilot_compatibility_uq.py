#!/usr/bin/env python3
"""CompatibilityUQ pilot with a parameter-matched strong baseline gate.

The direct four-way baseline can only use conclusion and reasoning features
additively.  CompatibilityUQ instead uses explicit nonlinear interactions
between their frozen hidden states, while retaining a linear four-way head and
nearly identical parameter count.  All runs use one fixed seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.pilot_typed_risk import load_split
from analysis.typeuq_baseline_gate import (
    DATASETS,
    FourWayLinear,
    aggregate,
    fit,
    joint_metrics,
    quadrant,
    seed_all,
)


HIDDEN_DIM = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("popllm/cached_features/5k"))
    parser.add_argument("--model-name", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--train-max-samples", type=int, default=5000)
    parser.add_argument("--ood-max-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--baseline-result",
        type=Path,
        default=Path("popllm/results/refactor/typeuq_baseline_gate_llama.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_compatibility_uq_llama.json"),
    )
    return parser.parse_args()


def compatibility_features(x: torch.Tensor) -> torch.Tensor:
    """Build interaction-only features from the two frozen representations."""
    half = x.shape[1] // 2
    if x.shape[1] % 2 or half <= HIDDEN_DIM:
        raise ValueError(f"Unexpected concatenated feature shape: {tuple(x.shape)}")
    conclusion_hidden = x[:, :HIDDEN_DIM]
    reasoning_hidden = x[:, half : half + HIDDEN_DIM]
    conclusion_aux = x[:, HIDDEN_DIM:half]
    reasoning_aux = x[:, half + HIDDEN_DIM :]

    # Unit normalization removes norm as a shortcut. sqrt(d) scaling keeps the
    # elementwise interactions numerically well-conditioned before z-scoring.
    scale = float(HIDDEN_DIM) ** 0.5
    conclusion_unit = torch.nn.functional.normalize(conclusion_hidden, dim=1)
    reasoning_unit = torch.nn.functional.normalize(reasoning_hidden, dim=1)
    hidden_product = conclusion_unit * reasoning_unit * scale
    hidden_distance = (conclusion_unit - reasoning_unit).abs() * scale
    cosine = (conclusion_unit * reasoning_unit).sum(dim=1, keepdim=True)

    # Auxiliary frozen features contain token confidence/statistics. Preserve
    # both marginals and their discrepancy, but do not expose raw hidden states.
    auxiliary = torch.cat(
        (conclusion_aux, reasoning_aux, (conclusion_aux - reasoning_aux).abs()), dim=1
    )
    return torch.cat((hidden_product, hidden_distance, cosine, auxiliary), dim=1)


def deltas_against_baselines(
    compatibility: Dict[str, dict], baseline_payload: dict, datasets: tuple[str, ...]
) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    compatibility_summary = aggregate({"compatibility": compatibility}, datasets)["compatibility"]
    baseline_key = "id_summary" if datasets == ("hotpot_qa",) else "ood_mean"
    for method in ("scalar_mlp_matched", "fourway_linear", "separate_experts"):
        baseline = baseline_payload[baseline_key][method]
        shared = sorted(set(compatibility_summary).intersection(baseline))
        output[method] = {
            metric: float(compatibility_summary[metric] - baseline[metric]) for metric in shared
        }
    return output


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    hotpot = args.cache_root / "hotpot_qa" / args.model_name
    train_raw = load_split(hotpot, "train", args.train_max_samples)
    validation_raw = load_split(hotpot, "validation", args.train_max_samples)
    train_x = compatibility_features(train_raw[0])
    validation_x = compatibility_features(validation_raw[0])
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    train = ((train_x - mean).div(std).to(device), train_raw[1].to(device))
    validation = (
        (validation_x - mean).div(std).to(device),
        validation_raw[1].to(device),
    )

    seed_all(args.seed)
    model = FourWayLinear(train[0].shape[1]).to(device)
    model = fit(
        model,
        lambda current, x, y: torch.nn.functional.cross_entropy(current(x), quadrant(y)),
        train,
        validation,
        args,
    )

    per_dataset: Dict[str, dict] = {}
    counts: Dict[str, int] = {}
    for dataset in DATASETS:
        cache_dir = args.cache_root / dataset / args.model_name
        if not (cache_dir / "test").is_dir():
            continue
        limit = args.train_max_samples if dataset == "hotpot_qa" else args.ood_max_samples
        x_raw, y_cpu = load_split(cache_dir, "test", limit)
        x = compatibility_features(x_raw)
        x = ((x - mean) / std).to(device)
        y = y_cpu.to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(x), dim=1).cpu().numpy()
        per_dataset[dataset] = joint_metrics(y, probabilities)
        counts[dataset] = len(y)
        del x_raw, x, y_cpu, y
        if device.type == "cuda":
            torch.cuda.empty_cache()

    id_datasets = ("hotpot_qa",)
    ood_datasets = tuple(dataset for dataset in DATASETS if dataset != "hotpot_qa")
    baseline_payload = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    if baseline_payload["seed"] != args.seed or baseline_payload["model_name"] != args.model_name:
        raise ValueError("Baseline result does not match the requested seed/model")

    payload = {
        "method": "CompatibilityUQ",
        "model_name": args.model_name,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_samples": len(train[1]),
        "validation_samples": len(validation[1]),
        "test_samples": counts,
        "input_dim": train[0].shape[1],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "feature_definition": (
            "normalized hidden product + absolute normalized hidden difference + cosine + "
            "conclusion/reasoning auxiliary features and their absolute difference"
        ),
        "per_dataset": per_dataset,
        "id_summary": aggregate({"compatibility": per_dataset}, id_datasets)["compatibility"],
        "ood_mean": aggregate({"compatibility": per_dataset}, ood_datasets)["compatibility"],
        "delta_id": deltas_against_baselines(per_dataset, baseline_payload, id_datasets),
        "delta_ood": deltas_against_baselines(per_dataset, baseline_payload, ood_datasets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
