#!/usr/bin/env python3
"""Evidence-grounded multi-trajectory consensus pilot.

All grounding signals are computed against the provided context, without gold
supporting-fact titles or a judge.  The experiment reuses the cached stochastic
trajectories from P12 and asks whether grounded consensus beats answer-only
self-consistency.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.pilot_multitrajectory_uq import (
    citations,
    content_tokens,
    correct,
    extract_answer,
    extract_reasoning,
    featurize,
    normalize_answer,
)
from data.hotpotqa import HotpotQADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-cache",
        type=Path,
        default=Path("popllm/results/refactor/multitrajectory_hotpot_llama_generations.json"),
    )
    parser.add_argument(
        "--dataset-path", type=Path, default=Path("popllm/datasets/hotpot_qa/hf_dataset")
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_grounded_consensus_uq_llama.json"),
    )
    return parser.parse_args()


def resolve_title(citation: str, title_map: Dict[str, set[str]]) -> str | None:
    normalized = normalize_answer(citation)
    if normalized in title_map:
        return normalized
    candidates = [
        title for title in title_map if normalized and (normalized in title or title in normalized)
    ]
    return max(candidates, key=len) if candidates else None


def trajectory_grounding(text: str, title_map: Dict[str, set[str]]) -> tuple[float, float, float]:
    reasoning = extract_reasoning(text)
    cited = list(citations(reasoning))
    if not cited:
        return 0.0, 0.0, 0.0
    resolved = [resolve_title(value, title_map) for value in cited]
    validity = float(np.mean([value is not None for value in resolved]))

    step_scores = []
    for line in reasoning.splitlines():
        line_citations = citations(line)
        if not line_citations:
            continue
        step_tokens = content_tokens(re.sub(r"\[[^\]]+\]", " ", line))
        for cited_title in line_citations:
            title = resolve_title(cited_title, title_map)
            if title is None or not step_tokens:
                step_scores.append(0.0)
            else:
                # Precision of the generated claim's content words with respect
                # to its cited passage; this rewards explicit evidence support.
                step_scores.append(len(step_tokens & title_map[title]) / len(step_tokens))
    support = float(np.mean(step_scores)) if step_scores else 0.0
    grounding = validity * (0.5 + 0.5 * support)
    return grounding, validity, support


def score(labels: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
    }


def bootstrap_delta(
    labels: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, repetitions: int, seed: int
) -> Dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        deltas.append(
            roc_auc_score(labels[indices], candidate[indices])
            - roc_auc_score(labels[indices], baseline[indices])
        )
    values = np.asarray(deltas)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(values, (0.025, 0.975)).tolist(),
        "probability_nonpositive": float((values <= 0).mean()),
    }


def main() -> None:
    args = parse_args()
    rows = json.loads(args.generation_cache.read_text(encoding="utf-8"))
    base_x, labels, base_diagnostics = featurize(rows)
    dataset = HotpotQADataset(
        split="test", max_samples=len(rows), dataset_path=str(args.dataset_path)
    )
    if [item["question"] for item in dataset.data] != [row["question"] for row in rows]:
        raise ValueError("Dataset order does not match the generation cache")

    grounding_features, diagnostics = [], []
    for row, item, base in zip(rows, dataset.data, base_diagnostics):
        titles = item.get("context", {}).get("title", [])
        sentences = item.get("context", {}).get("sentences", [])
        title_map = {
            normalize_answer(title): content_tokens(" ".join(text))
            for title, text in zip(titles, sentences)
        }
        context_normalized = normalize_answer(
            " ".join(title + " " + " ".join(text) for title, text in zip(titles, sentences))
        )
        modal_answer = base["modal_answer"]
        modal_trajectories = [
            trajectory for trajectory in row["trajectories"]
            if extract_answer(trajectory["text"]) == modal_answer
        ]
        per_trajectory = [trajectory_grounding(value["text"], title_map) for value in modal_trajectories]
        grounding = float(np.mean([value[0] for value in per_trajectory]))
        validity = float(np.mean([value[1] for value in per_trajectory]))
        support = float(np.mean([value[2] for value in per_trajectory]))
        answer_present = float(bool(modal_answer) and modal_answer in context_normalized)
        answer_occurrences = min(context_normalized.count(modal_answer), 5) / 5 if modal_answer else 0.0
        frequency = base["modal_frequency"]
        grounded_consensus = frequency * (0.5 + 0.5 * grounding)
        context_consensus = frequency * (0.5 + 0.5 * answer_present)
        full_consensus = frequency * math.sqrt(
            (0.5 + 0.5 * grounding) * (0.5 + 0.5 * answer_present)
        )
        grounding_features.append(
            [grounding, validity, support, answer_present, answer_occurrences,
             grounded_consensus, context_consensus, full_consensus]
        )
        diagnostics.append(
            {
                **base,
                "trajectory_grounding": grounding,
                "citation_validity": validity,
                "step_passage_support": support,
                "answer_present_in_context": answer_present,
                "answer_occurrence_score": answer_occurrences,
                "grounded_consensus": grounded_consensus,
                "context_consensus": context_consensus,
                "full_grounded_consensus": full_consensus,
            }
        )
    grounding_x = np.asarray(grounding_features, dtype=float)

    indices = np.arange(len(labels))
    train_index, test_index = train_test_split(
        indices, test_size=0.5, random_state=args.seed, stratify=labels
    )
    answer_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    )
    grounded_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    )
    answer_model.fit(base_x[train_index, :4], labels[train_index])
    grounded_model.fit(
        np.concatenate((base_x[train_index, :4], grounding_x[train_index, :5]), axis=1),
        labels[train_index],
    )
    methods = {
        "self_consistency": base_x[test_index, 0],
        "negative_answer_entropy": base_x[test_index, 1],
        "grounding_only": grounding_x[test_index, 0],
        "answer_context_support_only": grounding_x[test_index, 3],
        "fixed_grounded_consensus": grounding_x[test_index, 5],
        "fixed_context_consensus": grounding_x[test_index, 6],
        "fixed_full_grounded_consensus": grounding_x[test_index, 7],
        "learned_answer_only_stack": answer_model.predict_proba(base_x[test_index, :4])[:, 1],
        "learned_grounded_stack": grounded_model.predict_proba(
            np.concatenate((base_x[test_index, :4], grounding_x[test_index, :5]), axis=1)
        )[:, 1],
    }
    test_results = {name: score(labels[test_index], values) for name, values in methods.items()}

    full_methods = {
        "self_consistency": base_x[:, 0],
        "fixed_grounded_consensus": grounding_x[:, 5],
        "fixed_context_consensus": grounding_x[:, 6],
        "fixed_full_grounded_consensus": grounding_x[:, 7],
    }
    full_results = {name: score(labels, values) for name, values in full_methods.items()}
    paired_bootstrap = {
        name: bootstrap_delta(labels, values, base_x[:, 0], args.bootstrap, args.seed)
        for name, values in full_methods.items() if name != "self_consistency"
    }
    payload = {
        "experiment": "evidence-grounded multi-trajectory consensus",
        "seed": args.seed,
        "samples": len(labels),
        "train_samples": len(train_index),
        "test_samples": len(test_index),
        "correct_modal_answers": int(labels.sum()),
        "grounding_access": "input context only; no gold supporting facts or judge",
        "grounding_feature_order": [
            "trajectory_grounding", "citation_validity", "step_passage_support",
            "answer_present_in_context", "answer_occurrence_score", "grounded_consensus",
            "context_consensus", "full_grounded_consensus",
        ],
        "test_results": test_results,
        "full_fixed_results": full_results,
        "paired_bootstrap_auroc_delta_vs_self_consistency": paired_bootstrap,
        "mean_grounding_by_correctness": {
            str(label): grounding_x[labels == label].mean(axis=0).tolist() for label in (0, 1)
        },
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
