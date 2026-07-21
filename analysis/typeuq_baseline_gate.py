#!/usr/bin/env python3
"""Strong-baseline gate for the TypeUQ direction.

All learned methods receive identical frozen conclusion/reasoning features and
label access.  The experiment uses one fixed seed and evaluates HotpotQA ID plus
six OOD reasoning datasets.  Its purpose is to reject TypeUQ early if explicit
typed factorization cannot beat direct scalar and four-way predictors.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.pilot_typed_risk import load_split


DATASETS = ("hotpot_qa", "MuSiQue", "IIRC", "2WikiMultihopQA", "StrategyQA", "StepGame", "babi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("popllm/cached_features/5k"))
    parser.add_argument("--model-name", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--train-max-samples", type=int, default=5000)
    parser.add_argument("--ood-max-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("popllm/results/refactor/typeuq_baseline_gate_llama.json")
    )
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quadrant(y: torch.Tensor) -> torch.Tensor:
    return (2 * y[:, 0].long() + y[:, 1].long()).long()


class ScalarMatched(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        # Hidden width 3 gives ~25k parameters, matching conditional TypeUQ.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 3), torch.nn.GELU(), torch.nn.Linear(3, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class FourWayLinear(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SeparateExperts(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, 2)


class ConditionalTypeUQ(torch.nn.Module):
    """P(answer) P(process | answer) with an explicit four-state joint."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, 3)

    def log_joint(self, x: torch.Tensor) -> torch.Tensor:
        answer, process_y0, process_y1 = self.linear(x).unbind(dim=1)
        log_y1 = torch.nn.functional.logsigmoid(answer)
        log_y0 = torch.nn.functional.logsigmoid(-answer)
        log_z1_y0 = torch.nn.functional.logsigmoid(process_y0)
        log_z0_y0 = torch.nn.functional.logsigmoid(-process_y0)
        log_z1_y1 = torch.nn.functional.logsigmoid(process_y1)
        log_z0_y1 = torch.nn.functional.logsigmoid(-process_y1)
        return torch.stack(
            (
                log_y0 + log_z0_y0,
                log_y0 + log_z1_y0,
                log_y1 + log_z0_y1,
                log_y1 + log_z1_y1,
            ),
            dim=1,
        )


def fit(
    model: torch.nn.Module,
    loss_fn,
    train: Tuple[torch.Tensor, torch.Tensor],
    val: Tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
) -> torch.nn.Module:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, state, stale = float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, *train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model, *val).item())
        if val_loss < best - 1e-6:
            best = val_loss
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(state)
    return model


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (probabilities >= edges[idx]) & (probabilities <= edges[idx + 1])
        else:
            mask = (probabilities >= edges[idx]) & (probabilities < edges[idx + 1])
        if mask.any():
            value += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(value if total else math.nan)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_auroc": float(roc_auc_score(labels, probabilities)),
        f"{prefix}_auprc": float(average_precision_score(labels, probabilities)),
        f"{prefix}_brier": float(brier_score_loss(labels, probabilities)),
        f"{prefix}_ece": ece(labels, probabilities),
    }


def joint_metrics(y: torch.Tensor, probabilities: np.ndarray) -> Dict[str, float]:
    labels = y.cpu().numpy().astype(int)
    classes = 2 * labels[:, 0] + labels[:, 1]
    answer_p = probabilities[:, 2] + probabilities[:, 3]
    process_p = probabilities[:, 1] + probabilities[:, 3]
    good_p = probabilities[:, 3]
    output = {}
    output.update(binary_metrics(labels[:, 0], answer_p, "answer"))
    output.update(binary_metrics(labels[:, 1], process_p, "process"))
    output.update(binary_metrics((classes == 3).astype(int), good_p, "all_good"))
    output["quadrant_macro_f1"] = float(
        f1_score(classes, probabilities.argmax(axis=1), labels=[0, 1, 2, 3], average="macro", zero_division=0)
    )
    output["quadrant_nll"] = float(-np.log(np.clip(probabilities[np.arange(len(classes)), classes], 1e-9, 1)).mean())
    return output


@torch.no_grad()
def probabilities_for(
    name: str,
    model: torch.nn.Module | None,
    x: torch.Tensor,
    x_raw: torch.Tensor | None = None,
) -> np.ndarray:
    if name == "token_confidence":
        if x_raw is None:
            raise ValueError("token_confidence requires unnormalized features")
        # X is concat(conclusion mean, reasoning mean); max-logp is the first
        # token-prob feature after the 4096-dimensional hidden state.
        answer_p = torch.exp(x_raw[:, 4096]).clamp(1e-5, 1 - 1e-5)
        process_p = torch.exp(x_raw[:, x_raw.shape[1] // 2 + 4096]).clamp(1e-5, 1 - 1e-5)
        return torch.stack(
            ((1 - answer_p) * (1 - process_p), (1 - answer_p) * process_p,
             answer_p * (1 - process_p), answer_p * process_p), dim=1
        ).cpu().numpy()
    if name == "scalar_mlp_matched":
        good = torch.sigmoid(model(x))
        # Scalar baseline only supports the all-good task; placeholder joint
        # probabilities are intentionally not constructed.
        return good.cpu().numpy()
    if name == "fourway_linear":
        return torch.softmax(model(x), dim=1).cpu().numpy()
    if name == "separate_experts":
        answer_p, process_p = torch.sigmoid(model.linear(x)).unbind(dim=1)
        return torch.stack(
            ((1 - answer_p) * (1 - process_p), (1 - answer_p) * process_p,
             answer_p * (1 - process_p), answer_p * process_p), dim=1
        ).cpu().numpy()
    if name == "typeuq_conditional":
        return model.log_joint(x).exp().cpu().numpy()
    raise ValueError(name)


def aggregate(results: Dict[str, Dict[str, dict]], datasets) -> Dict[str, dict]:
    summary = {}
    for method in results:
        rows = [results[method][dataset] for dataset in datasets if dataset in results[method]]
        keys = sorted(set.intersection(*(set(row) for row in rows))) if rows else []
        summary[method] = {
            key: float(np.mean([row[key] for row in rows])) for key in keys if isinstance(rows[0][key], (int, float))
        }
    return summary


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    hotpot = args.cache_root / "hotpot_qa" / args.model_name
    train_cpu = load_split(hotpot, "train", args.train_max_samples)
    val_cpu = load_split(hotpot, "validation", args.train_max_samples)
    mean = train_cpu[0].mean(dim=0, keepdim=True)
    std = train_cpu[0].std(dim=0, keepdim=True).clamp_min(1e-5)

    def prepare(data):
        return ((data[0] - mean) / std).to(device), data[1].to(device)

    train, val = prepare(train_cpu), prepare(val_cpu)
    in_dim = train[0].shape[1]
    models: Dict[str, torch.nn.Module | None] = {"token_confidence": None}

    seed_all(args.seed)
    scalar = ScalarMatched(in_dim).to(device)
    scalar = fit(
        scalar,
        lambda model, x, y: torch.nn.functional.binary_cross_entropy_with_logits(
            model(x), (y[:, 0] * y[:, 1]).float()
        ),
        train,
        val,
        args,
    )
    models["scalar_mlp_matched"] = scalar

    seed_all(args.seed)
    fourway = FourWayLinear(in_dim).to(device)
    fourway = fit(
        fourway,
        lambda model, x, y: torch.nn.functional.cross_entropy(model(x), quadrant(y)),
        train,
        val,
        args,
    )
    models["fourway_linear"] = fourway

    seed_all(args.seed)
    experts = SeparateExperts(in_dim).to(device)
    experts = fit(
        experts,
        lambda model, x, y: torch.nn.functional.binary_cross_entropy_with_logits(model.linear(x), y),
        train,
        val,
        args,
    )
    models["separate_experts"] = experts

    seed_all(args.seed)
    typeuq = ConditionalTypeUQ(in_dim).to(device)
    typeuq = fit(
        typeuq,
        lambda model, x, y: torch.nn.functional.nll_loss(model.log_joint(x), quadrant(y)),
        train,
        val,
        args,
    )
    models["typeuq_conditional"] = typeuq

    results: Dict[str, Dict[str, dict]] = {name: {} for name in models}
    counts = {}
    for dataset in DATASETS:
        cache_dir = args.cache_root / dataset / args.model_name
        if not (cache_dir / "test").is_dir():
            continue
        limit = args.train_max_samples if dataset == "hotpot_qa" else args.ood_max_samples
        data_cpu = load_split(cache_dir, "test", limit)
        x_raw, y_cpu = data_cpu
        x = ((x_raw - mean) / std).to(device)
        y = y_cpu.to(device)
        counts[dataset] = len(y)
        for name, model in models.items():
            probabilities = probabilities_for(name, model, x, x_raw=x_raw)
            if name == "scalar_mlp_matched":
                labels = (y[:, 0] * y[:, 1]).cpu().numpy().astype(int)
                results[name][dataset] = binary_metrics(labels, probabilities, "all_good")
            else:
                results[name][dataset] = joint_metrics(y, probabilities)
        del x, y, x_raw, y_cpu
        torch.cuda.empty_cache()

    payload = {
        "model_name": args.model_name,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_samples": len(train[1]),
        "validation_samples": len(val[1]),
        "test_samples": counts,
        "parameters": {
            name: 0 if model is None else sum(parameter.numel() for parameter in model.parameters())
            for name, model in models.items()
        },
        "per_dataset": results,
        "id_summary": aggregate(results, ("hotpot_qa",)),
        "ood_mean": aggregate(results, tuple(dataset for dataset in DATASETS if dataset != "hotpot_qa")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
