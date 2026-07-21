#!/usr/bin/env python3
"""Self-generated contrastive answer likelihood pilot.

Unique answers sampled across trajectories become counterfactual candidates.
Each candidate is teacher-forced under the same context and question without a
reasoning prefix.  We compare the modal answer's likelihood against its sampled
alternatives and test whether the relative margin improves over answer counts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.pilot_counterfactual_uq import score_rows
from analysis.pilot_multitrajectory_uq import correct, entropy, extract_answer, normalize_answer
from data.hotpotqa import HotpotQADataset
from utils.prompting import build_chat_prompt_input, prompt_to_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("popllm/models/Llama-3.1-8B-Instruct"))
    parser.add_argument("--dataset-path", type=Path, default=Path("popllm/datasets/hotpot_qa/hf_dataset"))
    parser.add_argument(
        "--generation-cache",
        type=Path,
        default=Path("popllm/results/refactor/multitrajectory_hotpot_llama_generations.json"),
    )
    parser.add_argument(
        "--score-cache",
        type=Path,
        default=Path("popllm/results/refactor/contrastive_answer_scores_hotpot_llama.json"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_contrastive_answer_uq_llama.json"),
    )
    return parser.parse_args()


def raw_answer(text: str) -> str:
    matches = re.findall(
        r"(?:^|\n)\s*(?:conclusion|final answer|answer)\s*[:\-]\s*([^\n\r]+)", text or "", re.I
    )
    value = matches[-1] if matches else ((text or "").strip().splitlines()[-1] if (text or "").strip() else "")
    value = value.strip().strip(" .,:;!?\"'")
    return re.sub(r"\s+", " ", value)[:160]


def context_text(item: dict) -> str:
    blocks = []
    for title, sentences in zip(
        item.get("context", {}).get("title", []), item.get("context", {}).get("sentences", [])
    ):
        blocks.append(f"[{title}] {''.join(sentences)}")
    return "\n".join(blocks)


def build_score_rows(tokenizer, dataset, generation_rows, max_length: int) -> tuple[list[dict], list[dict]]:
    rows, metadata = [], []
    for example_idx, (generation, item) in enumerate(zip(generation_rows, dataset.data)):
        normalized = [extract_answer(value["text"]) for value in generation["trajectories"]]
        raw = [raw_answer(value["text"]) for value in generation["trajectories"]]
        counts = Counter(normalized)
        modal, _ = counts.most_common(1)[0]
        representatives = {}
        for normalized_answer, raw_value in zip(normalized, raw):
            if normalized_answer and normalized_answer not in representatives and raw_value:
                # Correctness and clustering operate on this canonical answer.
                # Score exactly the same object rather than an arbitrary first
                # trajectory's capitalization, punctuation, or trailing text.
                representatives[normalized_answer] = normalized_answer
        if modal not in representatives:
            representatives[modal] = modal

        prompt = build_chat_prompt_input(
            tokenizer,
            [
                {"role": "system", "content": dataset.system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_text(item)}\n\nQuestion: {item['question']}",
                },
            ],
            add_generation_prompt=True,
        )
        prompt_ids = prompt_to_token_ids(tokenizer, prompt)
        prefix_ids = tokenizer("Conclusion:", add_special_tokens=False)["input_ids"]
        candidates = []
        for candidate_idx, (answer, representative) in enumerate(representatives.items()):
            target_ids = tokenizer(" " + representative, add_special_tokens=False)["input_ids"]
            if not target_ids:
                continue
            room = max_length - len(prefix_ids) - len(target_ids)
            input_ids = prompt_ids[-room:] + prefix_ids + target_ids
            rows.append(
                {
                    "example_idx": example_idx,
                    "variant": answer,
                    "input_ids": input_ids,
                    "target_start": len(input_ids) - len(target_ids),
                }
            )
            candidates.append(answer)
        metadata.append(
            {
                "question": generation["question"],
                "ground_truth": generation["ground_truth"],
                "answers": normalized,
                "counts": dict(counts),
                "modal_answer": modal,
                "candidates": candidates,
            }
        )
    return rows, metadata


def compute_scores(args, generation_rows, dataset) -> tuple[Dict[int, Dict[str, float]], list[dict]]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows, metadata = build_score_rows(tokenizer, dataset, generation_rows, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    scores = score_rows(model, tokenizer, rows, args.batch_size)
    payload = {"metadata": metadata, "scores": {str(key): value for key, value in scores.items()}}
    args.score_cache.parent.mkdir(parents=True, exist_ok=True)
    args.score_cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return scores, metadata


def score(labels: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
    }


def bootstrap(labels, candidate, baseline, repetitions, seed):
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generation_rows = json.loads(args.generation_cache.read_text(encoding="utf-8"))
    dataset = HotpotQADataset(
        split="test", max_samples=len(generation_rows), dataset_path=str(args.dataset_path)
    )
    if args.score_cache.exists():
        cached = json.loads(args.score_cache.read_text(encoding="utf-8"))
        metadata = cached["metadata"]
        scores = {int(key): value for key, value in cached["scores"].items()}
    else:
        scores, metadata = compute_scores(args, generation_rows, dataset)

    features, labels, diagnostics = [], [], []
    for example_idx, item in enumerate(metadata):
        candidate_scores = scores.get(example_idx, {})
        modal = item["modal_answer"]
        if modal not in candidate_scores:
            continue
        counts = Counter(item["answers"])
        total = sum(counts.values())
        modal_frequency = counts[modal] / total
        answer_entropy = entropy(counts.values())
        modal_logp = candidate_scores[modal]
        alternatives = [value for answer, value in candidate_scores.items() if answer != modal]
        strongest_alternative = max(alternatives) if alternatives else modal_logp - 5.0
        margin = modal_logp - strongest_alternative
        answer_names = list(candidate_scores)
        logps = np.asarray([candidate_scores[name] for name in answer_names])
        logps = logps - logps.max()
        likelihood = np.exp(logps)
        contrastive_mass = float(likelihood[answer_names.index(modal)] / likelihood.sum())
        weighted = np.asarray([counts[name] for name in answer_names]) * likelihood
        weighted_vote_mass = float(weighted[answer_names.index(modal)] / weighted.sum())
        margin_probability = float(1 / (1 + math.exp(-margin)))
        joint = math.sqrt(modal_frequency * contrastive_mass)
        features.append(
            [modal_frequency, -answer_entropy, modal_logp, margin, contrastive_mass,
             weighted_vote_mass, margin_probability, joint]
        )
        label = correct(modal, item["ground_truth"])
        labels.append(label)
        diagnostics.append({**item, "correct": label, "candidate_logps": candidate_scores})
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)

    indices = np.arange(len(y))
    train_index, test_index = train_test_split(
        indices, test_size=0.5, random_state=args.seed, stratify=y
    )
    answer_only = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    ).fit(x[train_index, :3], y[train_index])
    contrastive = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    ).fit(x[train_index], y[train_index])
    test_methods = {
        "self_consistency": x[test_index, 0],
        "negative_answer_entropy": x[test_index, 1],
        "modal_direct_loglikelihood": x[test_index, 2],
        "contrastive_margin": x[test_index, 3],
        "contrastive_likelihood_mass": x[test_index, 4],
        "likelihood_weighted_vote": x[test_index, 5],
        "fixed_consensus_contrastive_joint": x[test_index, 7],
        "learned_answer_only_stack": answer_only.predict_proba(x[test_index, :3])[:, 1],
        "learned_contrastive_stack": contrastive.predict_proba(x[test_index])[:, 1],
    }
    full_methods = {
        "self_consistency": x[:, 0],
        "contrastive_margin": x[:, 3],
        "contrastive_likelihood_mass": x[:, 4],
        "likelihood_weighted_vote": x[:, 5],
        "fixed_consensus_contrastive_joint": x[:, 7],
    }
    strongest_baseline = full_methods["likelihood_weighted_vote"]
    payload = {
        "experiment": "self-generated contrastive answer likelihood",
        "seed": args.seed,
        "samples": len(y),
        "correct_modal_answers": int(y.sum()),
        "teacher_forced_sequences": int(sum(len(value) for value in scores.values())),
        "feature_order": [
            "modal_frequency", "negative_answer_entropy", "modal_direct_logp", "margin",
            "contrastive_mass", "likelihood_weighted_vote_mass", "margin_probability",
            "fixed_consensus_contrastive_joint",
        ],
        "test_results": {name: score(y[test_index], values) for name, values in test_methods.items()},
        "full_fixed_results": {name: score(y, values) for name, values in full_methods.items()},
        "paired_bootstrap_delta_vs_likelihood_weighted_vote": {
            name: bootstrap(y, values, strongest_baseline, args.bootstrap, args.seed)
            for name, values in full_methods.items() if name != "likelihood_weighted_vote"
        },
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
