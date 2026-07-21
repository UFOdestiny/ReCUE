#!/usr/bin/env python3
"""Leave-one-domain-out pilot for robust hallucination uncertainty.

The target is answer correctness, so the experiment does not depend on the
reasoning judge.  Every learned method uses the same frozen conclusion and
reasoning features and the same linear detector.  Only the training objective
changes: ERM, class-balanced ERM, domain GroupDRO, or domain-by-failure-type
GroupDRO.  One fixed seed is used throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.pilot_typed_risk import load_split
from analysis.typeuq_baseline_gate import DATASETS, ece, seed_all


METHODS = ("token_confidence", "erm", "balanced_erm", "domain_gdro", "type_gdro")


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
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_domain_robust_uq_llama.json"),
    )
    return parser.parse_args()


def split_source(
    x: torch.Tensor, y: torch.Tensor, validation_fraction: float, seed: int
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic class-stratified split of a source domain."""
    train_indices, validation_indices = [], []
    generator = torch.Generator().manual_seed(seed)
    labels = y[:, 0].long()
    for label in (0, 1):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_validation = max(1, round(len(indices) * validation_fraction))
        validation_indices.append(indices[:n_validation])
        train_indices.append(indices[n_validation:])
    train_index = torch.cat(train_indices)
    validation_index = torch.cat(validation_indices)
    return (x[train_index], labels[train_index]), (x[validation_index], labels[validation_index])


class LinearDetector(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def group_mean(losses: torch.Tensor, groups: torch.Tensor, n_groups: int) -> torch.Tensor:
    values = []
    for group in range(n_groups):
        mask = groups == group
        # All constructed source groups are nonempty; this guard keeps the
        # objective well-defined for unusually small command-line subsets.
        values.append(losses[mask].mean() if mask.any() else losses.new_zeros(()))
    return torch.stack(values)


@torch.no_grad()
def validation_objective(
    model: LinearDetector,
    data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    method: str,
    n_domains: int,
) -> float:
    x, y, domains = data
    losses = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y.float(), reduction="none")
    if method in ("erm", "balanced_erm"):
        return float(losses.mean().item())
    if method == "domain_gdro":
        return float(group_mean(losses, domains, n_domains).max().item())
    type_groups = 2 * domains + y
    return float(group_mean(losses, type_groups, 2 * n_domains).max().item())


def fit_detector(
    method: str,
    train: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    validation: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_domains: int,
    args: argparse.Namespace,
) -> LinearDetector:
    seed_all(args.seed)
    x, y, domains = train
    model = LinearDetector(x.shape[1]).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if method == "domain_gdro":
        train_groups, n_groups = domains, n_domains
    elif method == "type_gdro":
        train_groups, n_groups = 2 * domains + y, 2 * n_domains
    else:
        train_groups, n_groups = domains, n_domains
    q = torch.ones(n_groups, device=x.device) / n_groups
    class_counts = torch.bincount(y, minlength=2).float().clamp_min(1)
    class_weights = len(y) / (2 * class_counts)

    best, best_state, stale = float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x), y.float(), reduction="none"
        )
        if method == "erm":
            loss = losses.mean()
        elif method == "balanced_erm":
            loss = (losses * class_weights[y]).mean()
        else:
            per_group = group_mean(losses, train_groups, n_groups)
            with torch.no_grad():
                q.mul_(torch.exp(args.dro_eta * per_group.detach().clamp(max=20)))
                q.div_(q.sum())
            loss = torch.dot(q, per_group)
        loss.backward()
        optimizer.step()

        model.eval()
        score = validation_objective(model, validation, method, n_domains)
        if score < best - 1e-6:
            best = score
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {method}")
    model.load_state_dict(best_state)
    return model


def aurc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities)
    errors = 1 - labels[order]
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(cumulative_risk.mean())


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    order = np.argsort(-probabilities)
    keep_80 = max(1, int(0.8 * len(labels)))
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": ece(labels, probabilities),
        "aurc": aurc(labels, probabilities),
        "risk_at_80_coverage": float((1 - labels[order[:keep_80]]).mean()),
    }


def summarize(per_target: Dict[str, Dict[str, dict]]) -> Dict[str, dict]:
    summary = {}
    for method, target_results in per_target.items():
        rows = list(target_results.values())
        summary[method] = {}
        for metric in rows[0]:
            values = [row[metric] for row in rows]
            summary[method][f"mean_{metric}"] = float(np.mean(values))
            if metric in ("auroc", "auprc"):
                summary[method][f"worst_{metric}"] = float(np.min(values))
            else:
                summary[method][f"worst_{metric}"] = float(np.max(values))
    return summary


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    validation_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    test_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    available = []
    for domain in DATASETS:
        cache_dir = args.cache_root / domain / args.model_name
        if not (cache_dir / "validation").is_dir() or not (cache_dir / "test").is_dir():
            continue
        validation_cache[domain] = load_split(cache_dir, "validation", args.source_max_samples)
        test_cache[domain] = load_split(cache_dir, "test", args.test_max_samples)
        available.append(domain)
    if len(available) < 3:
        raise RuntimeError("At least three domains are needed for leave-one-domain-out evaluation")

    per_target: Dict[str, Dict[str, dict]] = {method: {} for method in METHODS}
    sample_counts: Dict[str, dict] = {}
    parameter_count = None
    for target_index, target in enumerate(available):
        source_domains = [domain for domain in available if domain != target]
        train_parts, validation_parts = [], []
        for source_index, source in enumerate(source_domains):
            train_part, validation_part = split_source(
                *validation_cache[source],
                args.validation_fraction,
                args.seed + 100 * target_index + source_index,
            )
            train_parts.append((train_part[0], train_part[1], source_index))
            validation_parts.append((validation_part[0], validation_part[1], source_index))

        train_x = torch.cat([part[0] for part in train_parts])
        train_y = torch.cat([part[1] for part in train_parts])
        train_domains = torch.cat(
            [torch.full((len(part[1]),), part[2], dtype=torch.long) for part in train_parts]
        )
        validation_x = torch.cat([part[0] for part in validation_parts])
        validation_y = torch.cat([part[1] for part in validation_parts])
        validation_domains = torch.cat(
            [torch.full((len(part[1]),), part[2], dtype=torch.long) for part in validation_parts]
        )

        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
        train = ((train_x - mean).div(std).to(device), train_y.to(device), train_domains.to(device))
        validation = (
            (validation_x - mean).div(std).to(device),
            validation_y.to(device),
            validation_domains.to(device),
        )
        test_x_raw, test_y_raw = test_cache[target]
        test_x = ((test_x_raw - mean) / std).to(device)
        labels = test_y_raw[:, 0].long().numpy()

        token_probabilities = torch.exp(test_x_raw[:, 4096]).clamp(1e-6, 1 - 1e-6).numpy()
        per_target["token_confidence"][target] = metrics(labels, token_probabilities)
        for method in METHODS[1:]:
            model = fit_detector(method, train, validation, len(source_domains), args)
            with torch.no_grad():
                probabilities = torch.sigmoid(model(test_x)).cpu().numpy()
            per_target[method][target] = metrics(labels, probabilities)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())

        sample_counts[target] = {
            "source_train": len(train_y),
            "source_validation": len(validation_y),
            "target_test": len(labels),
        }
        del train, validation, test_x
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"completed held-out domain: {target}", flush=True)

    payload = {
        "experiment": "leave-one-domain-out answer-correctness UQ",
        "model_name": args.model_name,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "domains": available,
        "sample_counts": sample_counts,
        "feature": "concat(conclusion mean feature, reasoning mean feature)",
        "target": "answer_correct (no reasoning-judge target)",
        "learned_parameters": parameter_count,
        "per_target": per_target,
        "summary": summarize(per_target),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
