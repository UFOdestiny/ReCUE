#!/usr/bin/env python3
"""Multi-trajectory answer/path disagreement pilot.

Generate K stochastic reasoning trajectories per question and test whether path
agreement adds correctness-ranking signal beyond answer self-consistency and
answer-cluster entropy.  Generation and the train/test split use one fixed seed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.hotpotqa import HotpotQADataset


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "states", "step", "that", "the", "therefore",
    "this", "to", "was", "were", "which", "who", "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("popllm/models/Llama-3.1-8B-Instruct"))
    parser.add_argument("--dataset-path", type=Path, default=Path("popllm/datasets/hotpot_qa/hf_dataset"))
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--generation-cache",
        type=Path,
        default=Path("popllm/results/refactor/multitrajectory_hotpot_llama_generations.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_multitrajectory_uq_llama.json"),
    )
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer(text: str) -> str:
    matches = re.findall(r"(?:^|\n)\s*(?:conclusion|final answer|answer)\s*[:\-]\s*([^\n\r]+)", text, re.I)
    if matches:
        return normalize_answer(matches[-1])
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return normalize_answer(lines[-1] if lines else "")


def extract_reasoning(text: str) -> str:
    return re.split(r"(?:^|\n)\s*(?:conclusion|final answer|answer)\s*[:\-]", text or "", maxsplit=1, flags=re.I)[0]


def correct(answer: str, ground_truth: str) -> int:
    answer = normalize_answer(answer)
    ground_truth = normalize_answer(ground_truth)
    if not answer or not ground_truth:
        return 0
    return int(answer == ground_truth or answer in ground_truth or ground_truth in answer)


def content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def citations(text: str) -> set[str]:
    return {normalize_answer(value) for value in re.findall(r"\[([^\[\]]+)\]", text or "") if value.strip()}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def mean_pairwise(sets: Sequence[set[str]]) -> float:
    if len(sets) < 2:
        return 1.0
    return float(np.mean([jaccard(sets[i], sets[j]) for i, j in combinations(range(len(sets)), 2)]))


def entropy(counts: Iterable[int]) -> float:
    values = np.asarray(list(counts), dtype=float)
    probabilities = values / values.sum()
    raw = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum()
    return float(raw / math.log(len(values))) if len(values) > 1 else 0.0


def generate(args: argparse.Namespace) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    dataset = HotpotQADataset(
        split="test", max_samples=args.samples, dataset_path=str(args.dataset_path)
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    prompts, metadata = [], []
    for item in dataset.data:
        messages = dataset.build_chat_messages(item)
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        metadata.append({"question": item["question"], "ground_truth": dataset.get_ground_truth(item)})

    llm = LLM(
        model=str(args.model_path),
        tokenizer=str(args.model_path),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.55,
        max_model_len=4096,
        seed=args.seed,
        trust_remote_code=True,
        dtype="auto",
        disable_log_stats=True,
        attention_backend="FLASHINFER",
    )
    sampling = SamplingParams(
        n=args.trajectories,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        logprobs=1,
        seed=args.seed,
    )
    requests = llm.generate(prompts, sampling, use_tqdm=True)
    rows = []
    for info, request in zip(metadata, requests):
        trajectories = []
        for output in request.outputs:
            token_count = max(1, len(output.token_ids))
            trajectories.append(
                {
                    "text": output.text,
                    "mean_logprob": float(output.cumulative_logprob / token_count),
                }
            )
        rows.append({**info, "trajectories": trajectories})
    args.generation_cache.parent.mkdir(parents=True, exist_ok=True)
    args.generation_cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def featurize(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    features, labels, diagnostics = [], [], []
    for row in rows:
        answers = [extract_answer(item["text"]) for item in row["trajectories"]]
        counts = Counter(answers)
        modal_answer, modal_count = counts.most_common(1)[0]
        modal_indices = [index for index, answer in enumerate(answers) if answer == modal_answer]
        modal_trajectories = [row["trajectories"][index] for index in modal_indices]
        reasoning = [extract_reasoning(item["text"]) for item in modal_trajectories]
        lexical_consistency = mean_pairwise([content_tokens(text) for text in reasoning])
        citation_consistency = mean_pairwise([citations(text) for text in reasoning])
        lengths = np.asarray([len(content_tokens(text)) for text in reasoning], dtype=float)
        length_cv = float(lengths.std() / max(1.0, lengths.mean()))
        modal_frequency = modal_count / len(answers)
        answer_entropy = entropy(counts.values())
        modal_logprob = float(np.mean([item["mean_logprob"] for item in modal_trajectories]))
        number_clusters = len(counts) / len(answers)
        fixed_joint = modal_frequency * (0.5 + 0.5 * lexical_consistency)
        feature = [
            modal_frequency,
            -answer_entropy,
            modal_logprob,
            -number_clusters,
            lexical_consistency,
            citation_consistency,
            -length_cv,
            fixed_joint,
        ]
        label = correct(modal_answer, row["ground_truth"])
        features.append(feature)
        labels.append(label)
        diagnostics.append(
            {
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "modal_answer": modal_answer,
                "correct": label,
                "modal_frequency": modal_frequency,
                "answer_entropy": answer_entropy,
                "modal_logprob": modal_logprob,
                "lexical_consistency": lexical_consistency,
                "citation_consistency": citation_consistency,
                "length_cv": length_cv,
                "fixed_joint": fixed_joint,
            }
        )
    return np.asarray(features, dtype=float), np.asarray(labels, dtype=int), diagnostics


def score(labels: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
    }


def main() -> None:
    args = parse_args()
    if args.generation_cache.exists():
        rows = json.loads(args.generation_cache.read_text(encoding="utf-8"))
        if len(rows) != args.samples or any(len(row["trajectories"]) != args.trajectories for row in rows):
            raise ValueError("Existing generation cache does not match --samples/--trajectories")
    else:
        rows = generate(args)
    x, y, diagnostics = featurize(rows)
    indices = np.arange(len(y))
    train_index, test_index = train_test_split(
        indices, test_size=0.5, random_state=args.seed, stratify=y
    )

    # Feature columns 0:4 are answer-only; 4:8 add path information.
    answer_model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed))
    path_model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed))
    answer_model.fit(x[train_index, :4], y[train_index])
    path_model.fit(x[train_index], y[train_index])
    answer_probability = answer_model.predict_proba(x[test_index, :4])[:, 1]
    path_probability = path_model.predict_proba(x[test_index])[:, 1]

    methods = {
        "self_consistency": x[test_index, 0],
        "negative_answer_entropy": x[test_index, 1],
        "modal_mean_logprob": x[test_index, 2],
        "fixed_answer_path_consistency": x[test_index, 7],
        "learned_answer_only_stack": answer_probability,
        "learned_answer_plus_path_stack": path_probability,
    }
    results = {name: score(y[test_index], values) for name, values in methods.items()}
    payload = {
        "experiment": "multi-trajectory answer/path disagreement",
        "model_path": str(args.model_path),
        "dataset": "HotpotQA test",
        "seed": args.seed,
        "samples": len(y),
        "trajectories_per_sample": args.trajectories,
        "correct_modal_answers": int(y.sum()),
        "train_samples": len(train_index),
        "test_samples": len(test_index),
        "feature_order": [
            "modal_frequency", "negative_answer_entropy", "modal_mean_logprob",
            "negative_cluster_fraction", "path_lexical_consistency", "citation_consistency",
            "negative_path_length_cv", "fixed_answer_path_consistency",
        ],
        "test_results": results,
        "mean_features_by_correctness": {
            str(label): x[y == label].mean(axis=0).tolist() for label in (0, 1)
        },
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
