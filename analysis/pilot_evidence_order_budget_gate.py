#!/usr/bin/env python3
"""Compute-matched gate for evidence-order invariance UQ.

The first K ordinary generations select one fixed modal answer.  At equal
additional cost, either another K ordinary generations or K passage-order
permutations probe support for that same answer.  This isolates the value of
the intervention from the value of simply drawing more samples.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.pilot_multitrajectory_uq import correct, extract_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary-k16-cache", type=Path, required=True)
    parser.add_argument("--permutation-cache", type=Path, required=True)
    parser.add_argument("--selection-samples", type=int, default=8)
    parser.add_argument("--probe-samples", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metrics(labels: np.ndarray, values: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
    }


def paired_bootstrap(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float | list[float]]]:
    rng = np.random.default_rng(seed)
    auroc_deltas, auprc_deltas = [], []
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        auroc_deltas.append(
            roc_auc_score(sampled_labels, candidate[indices])
            - roc_auc_score(sampled_labels, baseline[indices])
        )
        auprc_deltas.append(
            average_precision_score(sampled_labels, candidate[indices])
            - average_precision_score(sampled_labels, baseline[indices])
        )
    output = {}
    for name, deltas in (("auroc", auroc_deltas), ("auprc", auprc_deltas)):
        values = np.asarray(deltas)
        output[name] = {
            "mean": float(values.mean()),
            "ci95": np.quantile(values, (0.025, 0.975)).tolist(),
            "probability_nonpositive": float((values <= 0).mean()),
        }
    return output


def main() -> None:
    args = parse_args()
    ordinary = json.loads(args.ordinary_k16_cache.read_text(encoding="utf-8"))
    permuted = json.loads(args.permutation_cache.read_text(encoding="utf-8"))
    if len(ordinary) != len(permuted):
        raise ValueError("Ordinary and permutation cache sizes differ")

    k, probe_k = args.selection_samples, args.probe_samples
    labels, initial_sc, repeat_support, order_support = [], [], [], []
    labels_k16, sc_k16 = [], []
    diagnostics = []
    for ordinary_row, permutation_row in zip(ordinary, permuted):
        trajectories = ordinary_row["trajectories"]
        permutations = permutation_row["permutations"]
        if len(trajectories) < k + probe_k or len(permutations) < probe_k:
            raise ValueError("A row has fewer generations than requested")

        selection_answers = [extract_answer(item["text"]) for item in trajectories[:k]]
        selection_counts = Counter(selection_answers)
        target, target_count = selection_counts.most_common(1)[0]
        repeat_answers = [
            extract_answer(item["text"]) for item in trajectories[k : k + probe_k]
        ]
        permutation_answers = [
            extract_answer(item["text"]) for item in permutations[:probe_k]
        ]

        all_answers = selection_answers + repeat_answers
        all_counts = Counter(all_answers)
        modal_k16, modal_k16_count = all_counts.most_common(1)[0]

        label = correct(target, ordinary_row["ground_truth"])
        labels.append(label)
        initial_sc.append(target_count / k)
        repeat_support.append(repeat_answers.count(target) / probe_k)
        order_support.append(permutation_answers.count(target) / probe_k)
        labels_k16.append(correct(modal_k16, ordinary_row["ground_truth"]))
        sc_k16.append(modal_k16_count / (k + probe_k))
        diagnostics.append(
            {
                "question": ordinary_row["question"],
                "ground_truth": ordinary_row["ground_truth"],
                "selected_target": target,
                "target_correct": label,
                "initial_consistency": initial_sc[-1],
                "repeat_support": repeat_support[-1],
                "order_support": order_support[-1],
                "modal_k16": modal_k16,
                "modal_k16_correct": labels_k16[-1],
            }
        )

    y = np.asarray(labels, dtype=int)
    base = np.asarray(initial_sc, dtype=float)
    repeated = np.asarray(repeat_support, dtype=float)
    reordered = np.asarray(order_support, dtype=float)
    repeated_joint = np.sqrt(base * repeated)
    reordered_joint = np.sqrt(base * reordered)
    y_k16 = np.asarray(labels_k16, dtype=int)
    sc16 = np.asarray(sc_k16, dtype=float)

    same_target_methods = {
        "selection_self_consistency_k8": base,
        "additional_ordinary_support_k8": repeated,
        "evidence_order_support_k8": reordered,
        "repeated_sampling_joint_k16": repeated_joint,
        "evidence_order_joint_k16": reordered_joint,
    }
    payload = {
        "experiment": "compute-matched evidence-order invariance gate",
        "seed": args.seed,
        "samples": len(y),
        "selection_generations": k,
        "probe_generations": probe_k,
        "correct_selected_targets": int(y.sum()),
        "same_target_results": {
            name: metrics(y, values) for name, values in same_target_methods.items()
        },
        "standard_vanilla_k16": {
            "correct_modal_answers": int(y_k16.sum()),
            **metrics(y_k16, sc16),
        },
        "paired_candidate_minus_repeated_sampling": paired_bootstrap(
            y, reordered_joint, repeated_joint, args.bootstrap, args.seed
        ),
        "paired_order_support_minus_ordinary_support": paired_bootstrap(
            y, reordered, repeated, args.bootstrap, args.seed
        ),
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
