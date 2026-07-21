#!/usr/bin/env python3
"""Small-sample identifiability pilot for answer and process risks.

This is a diagnostic experiment, not the final TypeUQ implementation.  It uses
the existing frozen ChainUQ caches and compares a one-dimensional shared risk
bottleneck against two independent linear risk heads.  Judge annotations are
targets only; they are never included in the input features.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("popllm/cached_features/5k/hotpot_qa/Llama-3.1-8B-Instruct"),
    )
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_typed_risk.json"),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(cache_dir: Path, split: str, max_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    split_dir = cache_dir / split
    base_paths = sorted(
        path for path in split_dir.glob("chunk_*.pt") if not path.name.endswith("_reasoning.pt")
    )
    xs, ys = [], []
    remaining = max_samples if max_samples > 0 else None
    for base_path in base_paths:
        sidecar_path = base_path.with_name(f"{base_path.stem}_reasoning.pt")
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        reasoning = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        take = len(base) if remaining is None else min(len(base), remaining)
        for sample, process in zip(base[:take], reasoning[:take]):
            n_steps = int(sample.get("n_reasoning_claims", 0) or 0)
            n_correct = int(sample.get("n_reasoning_correct", 0) or 0)
            label = sample.get("label", -1)
            token_features = process.get("reasoning_features")
            token_mask = process.get("reasoning_attention_mask")
            if label not in (0, 1) or n_steps <= 0 or not torch.is_tensor(token_features):
                continue
            if torch.is_tensor(token_mask):
                valid = token_mask.bool()
                token_features = token_features[valid]
            if token_features.numel() == 0:
                continue
            conclusion = sample["features"].float().mean(dim=0)
            reason_mean = token_features.float().mean(dim=0)
            # Keep the representation deliberately simple: conclusion state and
            # mean reasoning state, with no learned pooling or judge features.
            xs.append(torch.cat((conclusion, reason_mean), dim=0))
            ys.append((float(label), float(n_correct == n_steps)))
        if remaining is not None:
            remaining -= take
            if remaining <= 0:
                break
    if not xs:
        raise RuntimeError(f"No usable examples in {split_dir}")
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32)


class SharedScalar(torch.nn.Module):
    """Both risks must be affine functions of one learned scalar direction."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.scalar = torch.nn.Linear(in_dim, 1)
        self.readout = torch.nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.scalar(x))


class TypedLinear(torch.nn.Module):
    """Independent answer-risk and process-risk directions."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.heads = torch.nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.heads(x)


@torch.no_grad()
def loss_on(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.nn.functional.binary_cross_entropy_with_logits(model(x), y).item())


def fit(
    model: torch.nn.Module,
    train: Tuple[torch.Tensor, torch.Tensor],
    val: Tuple[torch.Tensor, torch.Tensor],
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> torch.nn.Module:
    x_train, y_train = train
    x_val, y_val = val
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        val_loss = loss_on(model, x_val, y_val)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    probabilities = torch.sigmoid(model(x)).float().cpu().numpy()
    labels = y.float().cpu().numpy().astype(int)
    predictions = (probabilities >= 0.5).astype(int)
    names = ("answer", "process")
    result: Dict[str, float] = {}
    for idx, name in enumerate(names):
        result[f"{name}_auroc"] = float(roc_auc_score(labels[:, idx], probabilities[:, idx]))
        result[f"{name}_auprc"] = float(average_precision_score(labels[:, idx], probabilities[:, idx]))
        result[f"{name}_brier"] = float(brier_score_loss(labels[:, idx], probabilities[:, idx]))
    true_quadrant = labels[:, 0] * 2 + labels[:, 1]
    pred_quadrant = predictions[:, 0] * 2 + predictions[:, 1]
    result["quadrant_macro_f1"] = float(
        f1_score(true_quadrant, pred_quadrant, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    )
    result["two_axis_exact_match"] = float(np.mean(np.all(predictions == labels, axis=1)))
    result["mean_auroc"] = (result["answer_auroc"] + result["process_auroc"]) / 2
    return result


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)

    splits = {
        split: load_split(args.cache_dir, split, args.max_samples)
        for split in ("train", "validation", "test")
    }
    mean = splits["train"][0].mean(dim=0, keepdim=True)
    std = splits["train"][0].std(dim=0, keepdim=True).clamp_min(1e-5)
    for split, (x, y) in splits.items():
        splits[split] = (((x - mean) / std).to(device), y.to(device))

    input_dim = splits["train"][0].shape[1]
    models = {
        "shared_scalar": SharedScalar(input_dim),
        "typed_linear": TypedLinear(input_dim),
    }
    results: Dict[str, object] = {
        "cache_dir": str(args.cache_dir),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "samples": {split: len(data[1]) for split, data in splits.items()},
        "targets": ["answer_correct", "all_reasoning_steps_valid"],
        "input": "concat(conclusion_feature, mean(reasoning_token_features))",
        "models": {},
    }
    for name, model in models.items():
        seed_everything(args.seed)
        model = model.to(device)
        model = fit(
            model,
            splits["train"],
            splits["validation"],
            args.epochs,
            args.patience,
            args.lr,
            args.weight_decay,
        )
        results["models"][name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "validation": evaluate(model, *splits["validation"]),
            "test": evaluate(model, *splits["test"]),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
