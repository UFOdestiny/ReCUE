#!/usr/bin/env python3
"""Contextual Evidence Gain (CEG) uncertainty pilot.

For each sampled candidate answer, contrast its teacher-forced likelihood under
the full QA context with an empty-context prior.  No generated reasoning prefix
is retained, preventing the model's own rationale from leaking the answer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.pilot_contrastive_answer_uq import bootstrap, score
from analysis.pilot_counterfactual_uq import score_rows
from analysis.pilot_multitrajectory_uq import correct
from data.hotpotqa import HotpotQADataset
from utils.prompting import build_chat_prompt_input, prompt_to_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--dataset-path", type=Path, default=Path("popllm/datasets/hotpot_qa/hf_dataset")
    )
    parser.add_argument("--generation-cache", type=Path, required=True)
    parser.add_argument("--full-score-cache", type=Path, required=True)
    parser.add_argument("--prior-score-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stable_mass(names: list[str], values: dict[str, float], target: str) -> float:
    array = np.asarray([values[name] for name in names], dtype=float)
    weights = np.exp(array - array.max())
    return float(weights[names.index(target)] / weights.sum())


def stable_weighted_mass(
    names: list[str], values: dict[str, float], counts: Counter, target: str
) -> float:
    array = np.asarray([values[name] for name in names], dtype=float)
    weights = np.exp(array - array.max()) * np.asarray([counts[name] for name in names])
    return float(weights[names.index(target)] / weights.sum())


def build_prior_rows(tokenizer, dataset, metadata: list[dict], max_length: int) -> list[dict]:
    rows = []
    for example_idx, (item, datum) in enumerate(zip(metadata, dataset.data)):
        prompt = build_chat_prompt_input(
            tokenizer,
            [
                {"role": "system", "content": dataset.system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n[No context provided]\n\nQuestion: {datum['question']}",
                },
            ],
            add_generation_prompt=True,
        )
        prompt_ids = prompt_to_token_ids(tokenizer, prompt)
        prefix_ids = tokenizer("Conclusion:", add_special_tokens=False)["input_ids"]
        for candidate in item["candidates"]:
            target_ids = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
            if not target_ids:
                continue
            room = max_length - len(prefix_ids) - len(target_ids)
            input_ids = prompt_ids[-room:] + prefix_ids + target_ids
            rows.append(
                {
                    "example_idx": example_idx,
                    "variant": candidate,
                    "input_ids": input_ids,
                    "target_start": len(input_ids) - len(target_ids),
                }
            )
    return rows


def compute_prior_scores(args, metadata: list[dict], dataset) -> dict[int, dict[str, float]]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = build_prior_rows(tokenizer, dataset, metadata, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    values = score_rows(model, tokenizer, rows, args.batch_size)
    args.prior_score_cache.parent.mkdir(parents=True, exist_ok=True)
    args.prior_score_cache.write_text(
        json.dumps({"scores": {str(key): value for key, value in values.items()}}, indent=2),
        encoding="utf-8",
    )
    return values


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    generation = json.loads(args.generation_cache.read_text(encoding="utf-8"))
    full_cache = json.loads(args.full_score_cache.read_text(encoding="utf-8"))
    metadata = full_cache["metadata"]
    full_scores = {int(key): value for key, value in full_cache["scores"].items()}
    dataset = HotpotQADataset(
        split="test", max_samples=len(generation), dataset_path=str(args.dataset_path)
    )
    if args.prior_score_cache.exists():
        cached = json.loads(args.prior_score_cache.read_text(encoding="utf-8"))
        prior_scores = {int(key): value for key, value in cached["scores"].items()}
    else:
        prior_scores = compute_prior_scores(args, metadata, dataset)

    features, labels, diagnostics = [], [], []
    for index, item in enumerate(metadata):
        modal = item["modal_answer"]
        full = full_scores.get(index, {})
        prior = prior_scores.get(index, {})
        names = [name for name in item["candidates"] if name in full and name in prior]
        if modal not in names:
            continue
        counts = Counter(item["answers"])
        sc = counts[modal] / sum(counts.values())
        gain = {name: full[name] - prior[name] for name in names}
        full_mass = stable_mass(names, full, modal)
        full_weighted = stable_weighted_mass(names, full, counts, modal)
        prior_mass = stable_mass(names, prior, modal)
        evidence_mass = stable_mass(names, gain, modal)
        evidence_weighted = stable_weighted_mass(names, gain, counts, modal)
        fixed_joint = math.sqrt(sc * evidence_mass)
        fixed_weighted_joint = math.sqrt(sc * evidence_weighted)
        label = correct(modal, item["ground_truth"])
        features.append(
            [
                sc,
                full[modal],
                prior[modal],
                gain[modal],
                full_mass,
                full_weighted,
                prior_mass,
                evidence_mass,
                evidence_weighted,
                fixed_joint,
                fixed_weighted_joint,
            ]
        )
        labels.append(label)
        diagnostics.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "modal_answer": modal,
                "correct": label,
                "candidate_full_logp": full,
                "candidate_prior_logp": prior,
                "candidate_context_gain": gain,
            }
        )

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    methods = {
        "self_consistency": x[:, 0],
        "full_context_modal_logp": x[:, 1],
        "modal_context_gain": x[:, 3],
        "full_contrastive_mass": x[:, 4],
        "full_likelihood_weighted_vote": x[:, 5],
        "prior_contrastive_mass": x[:, 6],
        "evidence_bayes_factor_mass": x[:, 7],
        "evidence_bayes_factor_weighted_vote": x[:, 8],
        "fixed_sc_evidence_joint": x[:, 9],
        "fixed_sc_weighted_evidence_joint": x[:, 10],
    }
    candidate_names = (
        "modal_context_gain",
        "evidence_bayes_factor_mass",
        "evidence_bayes_factor_weighted_vote",
        "fixed_sc_evidence_joint",
        "fixed_sc_weighted_evidence_joint",
    )
    baseline_names = (
        "self_consistency",
        "full_context_modal_logp",
        "full_contrastive_mass",
        "full_likelihood_weighted_vote",
    )
    payload = {
        "experiment": "contextual evidence gain UQ",
        "model_path": str(args.model_path),
        "seed": args.seed,
        "samples": len(y),
        "correct_modal_answers": int(y.sum()),
        "feature_order": [
            "self_consistency",
            "full_modal_logp",
            "prior_modal_logp",
            "modal_context_gain",
            "full_mass",
            "full_weighted_vote",
            "prior_mass",
            "evidence_mass",
            "evidence_weighted_vote",
            "fixed_joint",
            "fixed_weighted_joint",
        ],
        "full_results": {name: score(y, values) for name, values in methods.items()},
        "paired_bootstrap_candidate_vs_each_baseline": {
            candidate: {
                baseline: bootstrap(
                    y, methods[candidate], methods[baseline], args.bootstrap, args.seed
                )
                for baseline in baseline_names
            }
            for candidate in candidate_names
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
