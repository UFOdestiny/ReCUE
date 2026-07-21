#!/usr/bin/env python3
"""Unlabeled target-domain normalization pilot for hallucination UQ.

This tests whether cross-domain detector failure is primarily caused by feature
location/scale drift.  Source domains and the unlabeled target batch are each
standardized with their own moments, then the exact same linear ERM detector is
trained.  No target labels are used for fitting or model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.pilot_domain_robust_uq import fit_detector, metrics, split_source, summarize
from analysis.pilot_typed_risk import load_split
from analysis.typeuq_baseline_gate import DATASETS, seed_all


METHODS = ("domain_relative_erm", "domain_relative_balanced_erm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("popllm/cached_features/5k"))
    parser.add_argument("--model-name", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--source-max-samples", type=int, default=1500)
    parser.add_argument("--test-max-samples", type=int, default=1500)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dro-eta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--baseline-result",
        type=Path,
        default=Path("popllm/results/refactor/pilot_domain_robust_uq_llama.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_domain_relative_uq_llama.json"),
    )
    return parser.parse_args()


def standardize(
    reference: torch.Tensor, *values: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    mean = reference.mean(dim=0, keepdim=True)
    std = reference.std(dim=0, keepdim=True).clamp_min(1e-5)
    return tuple((value - mean) / std for value in values)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    validation_cache = {}
    test_cache = {}
    available = []
    for domain in DATASETS:
        cache_dir = args.cache_root / domain / args.model_name
        if not (cache_dir / "validation").is_dir() or not (cache_dir / "test").is_dir():
            continue
        validation_cache[domain] = load_split(cache_dir, "validation", args.source_max_samples)
        test_cache[domain] = load_split(cache_dir, "test", args.test_max_samples)
        available.append(domain)

    per_target: Dict[str, Dict[str, dict]] = {method: {} for method in METHODS}
    counts = {}
    for target_index, target in enumerate(available):
        source_domains = [domain for domain in available if domain != target]
        train_parts, validation_parts = [], []
        for source_index, source in enumerate(source_domains):
            train_part, validation_part = split_source(
                *validation_cache[source],
                args.validation_fraction,
                args.seed + 100 * target_index + source_index,
            )
            train_x, validation_x = standardize(train_part[0], train_part[0], validation_part[0])
            train_parts.append((train_x, train_part[1], source_index))
            validation_parts.append((validation_x, validation_part[1], source_index))

        train_x = torch.cat([part[0] for part in train_parts]).to(device)
        train_y = torch.cat([part[1] for part in train_parts]).to(device)
        train_domains = torch.cat(
            [torch.full((len(part[1]),), part[2], dtype=torch.long) for part in train_parts]
        ).to(device)
        validation_x = torch.cat([part[0] for part in validation_parts]).to(device)
        validation_y = torch.cat([part[1] for part in validation_parts]).to(device)
        validation_domains = torch.cat(
            [torch.full((len(part[1]),), part[2], dtype=torch.long) for part in validation_parts]
        ).to(device)
        train = (train_x, train_y, train_domains)
        validation = (validation_x, validation_y, validation_domains)

        test_x_raw, test_y_raw = test_cache[target]
        # This is the only test-time adaptation: moments of the unlabeled target
        # batch. Target correctness labels remain untouched until evaluation.
        (test_x,) = standardize(test_x_raw, test_x_raw)
        test_x = test_x.to(device)
        labels = test_y_raw[:, 0].long().numpy()
        for method in METHODS:
            training_method = "balanced_erm" if method.endswith("balanced_erm") else "erm"
            model = fit_detector(training_method, train, validation, len(source_domains), args)
            with torch.no_grad():
                probabilities = torch.sigmoid(model(test_x)).cpu().numpy()
            per_target[method][target] = metrics(labels, probabilities)
        counts[target] = len(labels)
        print(f"completed held-out domain: {target}", flush=True)

    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    learned_summary = summarize(per_target)
    comparison = {
        method: {
            baseline_method: {
                metric: learned_summary[method][metric] - baseline["summary"][baseline_method][metric]
                for metric in learned_summary[method]
            }
            for baseline_method in ("erm", "balanced_erm")
        }
        for method in METHODS
    }
    payload = {
        "experiment": "leave-one-domain-out unlabeled domain-relative normalization",
        "model_name": args.model_name,
        "seed": args.seed,
        "domains": available,
        "target": "answer_correct",
        "target_label_access": "none during target normalization/training/model selection",
        "test_samples": counts,
        "per_target": per_target,
        "summary": learned_summary,
        "delta_against_global_normalization": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
