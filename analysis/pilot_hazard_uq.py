#!/usr/bin/env python3
"""Step-wise error-hazard pilot on existing ChainUQ reasoning caches.

The learned score at step k only observes that step, its change from step k-1,
and normalized position.  It therefore supports online error warning.  Judge
verdicts are training/evaluation targets and never input features.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class SplitData:
    current: torch.Tensor
    delta: torch.Tensor
    position: torch.Tensor
    valid: torch.Tensor
    sample_index: torch.Tensor
    answer: torch.Tensor
    token_confidence: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("popllm/cached_features/5k/hotpot_qa/Llama-3.1-8B-Instruct"),
    )
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_hazard_uq.json"),
    )
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(cache_dir: Path, split: str, max_samples: int) -> SplitData:
    current, delta, position, valid, sample_index, token_confidence = [], [], [], [], [], []
    answers = []
    seen_samples = 0
    base_paths = sorted(
        path for path in (cache_dir / split).glob("chunk_*.pt")
        if not path.name.endswith("_reasoning.pt")
    )
    for base_path in base_paths:
        side_path = base_path.with_name(f"{base_path.stem}_reasoning.pt")
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        side = torch.load(side_path, map_location="cpu", weights_only=False)
        for sample, reasoning in zip(base, side):
            if max_samples > 0 and seen_samples >= max_samples:
                break
            labels = reasoning.get("reasoning_verified") or []
            claims = reasoning.get("reasoning_claims") or []
            features = reasoning.get("reasoning_features")
            if sample.get("label") not in (0, 1) or not torch.is_tensor(features):
                continue
            if len(labels) == 0 or len(labels) != len(claims) or any(label not in (0, 1) for label in labels):
                continue
            pooled: List[torch.Tensor] = []
            confidences: List[float] = []
            usable = True
            for claim in claims:
                ids = [int(idx) for idx in claim.get("aligned_token_ids", [])]
                ids = [idx for idx in ids if 0 <= idx < len(features)]
                if not ids:
                    usable = False
                    break
                step_tokens = features[ids].float()
                pooled.append(step_tokens.mean(dim=0))
                # Feature layout is hidden[4096] + top6 log-probs + four stats
                # + attention.  Higher mean max-logp means lower token uncertainty.
                confidences.append(float(step_tokens[:, 4096].mean().item()))
            if not usable:
                continue
            sid = len(answers)
            answers.append(float(sample["label"]))
            for idx, (step, label, conf) in enumerate(zip(pooled, labels, confidences)):
                previous = pooled[idx - 1] if idx > 0 else torch.zeros_like(step)
                current.append(step)
                delta.append(step - previous)
                position.append((idx + 1) / len(pooled))
                valid.append(float(label))
                sample_index.append(sid)
                token_confidence.append(conf)
            seen_samples += 1
        if max_samples > 0 and seen_samples >= max_samples:
            break
    if not current:
        raise RuntimeError(f"No usable step data in {cache_dir / split}")
    return SplitData(
        current=torch.stack(current),
        delta=torch.stack(delta),
        position=torch.tensor(position, dtype=torch.float32).unsqueeze(1),
        valid=torch.tensor(valid, dtype=torch.float32),
        sample_index=torch.tensor(sample_index, dtype=torch.long),
        answer=torch.tensor(answers, dtype=torch.float32),
        token_confidence=torch.tensor(token_confidence, dtype=torch.float32),
    )


def features(data: SplitData, variant: str) -> torch.Tensor:
    if variant == "current_linear":
        return data.current
    if variant == "hazard_delta":
        return torch.cat((data.current, data.delta, data.position), dim=1)
    raise ValueError(variant)


def fit_linear(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
) -> torch.nn.Module:
    model = torch.nn.Linear(x_train.shape[1], 1).to(x_train.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Balance rare invalid steps while retaining a probabilistic binary loss.
    invalid = (y_train == 0).sum().clamp_min(1)
    valid = (y_train == 1).sum().clamp_min(1)
    valid_weight = (invalid / valid).clamp_min(0.05)
    weights = torch.where(y_train > 0.5, valid_weight, torch.ones_like(y_train))
    best, state, stale = float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train, weight=weights)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(x_val).squeeze(1), y_val
            ).item()
        if val_loss < best - 1e-5:
            best = val_loss
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if state is None:
        raise RuntimeError("No checkpoint")
    model.load_state_dict(state)
    return model


def response_metrics(data: SplitData, step_scores: np.ndarray) -> Dict[str, float]:
    valid = data.valid.numpy().astype(int)
    sample_ids = data.sample_index.numpy()
    answer = data.answer.numpy().astype(int)
    unique = np.arange(len(answer))
    process_labels, survival, min_scores = [], [], []
    first_error_hits, flawed = 0, 0
    for sid in unique:
        mask = sample_ids == sid
        labels = valid[mask]
        scores = np.clip(step_scores[mask], 1e-6, 1 - 1e-6)
        process_labels.append(int(labels.all()))
        survival.append(float(np.exp(np.log(scores).sum())))
        min_scores.append(float(scores.min()))
        if not labels.all():
            flawed += 1
            first_error_hits += int(int(np.argmin(scores)) == int(np.flatnonzero(labels == 0)[0]))
    process_labels = np.asarray(process_labels)
    survival = np.asarray(survival)
    min_scores = np.asarray(min_scores)
    result = {
        "step_valid_auroc": float(roc_auc_score(valid, step_scores)),
        "step_invalid_auprc": float(average_precision_score(1 - valid, 1 - step_scores)),
        "first_error_top1": float(first_error_hits / max(flawed, 1)),
        "flawed_chains": int(flawed),
        "all_steps_valid_auroc_survival": float(roc_auc_score(process_labels, survival)),
        "all_steps_valid_auroc_min": float(roc_auc_score(process_labels, min_scores)),
        "answer_auroc_survival": float(roc_auc_score(answer, survival)),
    }
    return result


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    raw = {split: load_split(args.cache_dir, split, args.max_samples) for split in ("train", "validation", "test")}
    results: Dict[str, object] = {
        "cache_dir": str(args.cache_dir),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed,
        "samples": {split: len(data.answer) for split, data in raw.items()},
        "steps": {split: len(data.valid) for split, data in raw.items()},
        "invalid_step_rate": {split: float(1 - data.valid.mean()) for split, data in raw.items()},
        "models": {},
    }
    # Natural training-free baseline from token maximum log-probability.
    results["models"]["token_max_logp"] = response_metrics(
        raw["test"], torch.sigmoid(raw["test"].token_confidence).numpy()
    )
    for variant in ("current_linear", "hazard_delta"):
        xs = {split: features(data, variant) for split, data in raw.items()}
        mean = xs["train"].mean(dim=0, keepdim=True)
        std = xs["train"].std(dim=0, keepdim=True).clamp_min(1e-5)
        xs = {split: ((x - mean) / std).to(device) for split, x in xs.items()}
        ys = {split: data.valid.to(device) for split, data in raw.items()}
        seed_all(args.seed)
        model = fit_linear(xs["train"], ys["train"], xs["validation"], ys["validation"], args)
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(xs["test"]).squeeze(1)).float().cpu().numpy()
        results["models"][variant] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            **response_metrics(raw["test"], scores),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
