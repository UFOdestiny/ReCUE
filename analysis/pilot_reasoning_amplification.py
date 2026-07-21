#!/usr/bin/env python3
"""Pilot for hallucination detection via reasoning-induced confidence gain.

The same cached conclusion is teacher-forced twice under the original context:
once directly after ``Conclusion:`` and once after the model's own generated
reasoning prefix.  The pre-defined score ``pre_logp - post_logp`` penalizes
answers whose confidence is disproportionately amplified by their rationale.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.hotpotqa import HotpotQADataset
from utils.prompting import build_chat_prompt_input, prompt_to_token_ids
from analysis.pilot_counterfactual_uq import (
    balanced_samples,
    clean_answer,
    context_variants,
    first_chunk,
    messages,
    reasoning_prefix,
    score_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("popllm/models/Llama-3.1-8B-Instruct"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("popllm/cached_features/5k/hotpot_qa/Llama-3.1-8B-Instruct"),
    )
    parser.add_argument("--dataset-path", type=Path, default=Path("popllm/datasets/hotpot_qa/hf_dataset"))
    parser.add_argument("--train-per-class", type=int, default=32)
    parser.add_argument("--test-per-class", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--max-prefix-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_reasoning_amplification_llama.json"),
    )
    return parser.parse_args()


def make_rows(tokenizer, dataset, samples, max_length: int, max_prefix_tokens: int):
    rows, labels = [], []
    for example_idx, sample in enumerate(samples):
        raw = dataset.data[int(sample["sample_id"])]
        answer = clean_answer(sample)
        target_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
        prompt = build_chat_prompt_input(
            tokenizer,
            messages(dataset.system_prompt, raw["question"], context_variants(raw)["full"]),
            add_generation_prompt=True,
        )
        prompt_ids = prompt_to_token_ids(tokenizer, prompt)
        prefixes = {
            "pre": tokenizer("Conclusion:", add_special_tokens=False)["input_ids"],
            "post": tokenizer(reasoning_prefix(sample), add_special_tokens=False)["input_ids"][-max_prefix_tokens:],
        }
        for variant, prefix_ids in prefixes.items():
            room = max_length - len(prefix_ids) - len(target_ids)
            if room <= 32:
                continue
            clipped_prompt = prompt_ids[-room:]
            rows.append(
                {
                    "example_idx": example_idx,
                    "variant": variant,
                    "input_ids": clipped_prompt + prefix_ids + target_ids,
                    "target_start": len(clipped_prompt) + len(prefix_ids),
                    "answer": answer,
                }
            )
        labels.append(int(sample["label"]))
    return rows, labels


def matrix(scores, labels):
    x, y = [], []
    for idx, label in enumerate(labels):
        row = scores.get(idx, {})
        if "pre" not in row or "post" not in row:
            continue
        pre, post = row["pre"], row["post"]
        x.append([pre, post, post - pre, pre - post])
        y.append(label)
    return np.asarray(x), np.asarray(y, dtype=int)


def score(y, values):
    return {
        "auroc": float(roc_auc_score(y, values)),
        "auprc": float(average_precision_score(y, values)),
    }


def learned_curve(x_train, y_train, x_test, y_test):
    scaler = StandardScaler().fit(x_train[:, :3])
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=2026).fit(
        scaler.transform(x_train[:, :3]), y_train
    )
    probability = model.predict_proba(scaler.transform(x_test[:, :3]))[:, 1]
    return score(y_test, probability)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    selected = {
        "train": balanced_samples(first_chunk(args.cache_dir, "train"), args.train_per_class, args.seed),
        "test": balanced_samples(first_chunk(args.cache_dir, "test"), args.test_per_class, args.seed + 1),
    }
    datasets = {
        "train": HotpotQADataset(split="train", max_samples=5000, dataset_path=str(args.dataset_path)),
        "test": HotpotQADataset(split="test", max_samples=0, dataset_path=str(args.dataset_path)),
    }
    matrices = {}
    sequence_count = 0
    for split in ("train", "test"):
        rows, labels = make_rows(
            tokenizer, datasets[split], selected[split], args.max_length, args.max_prefix_tokens
        )
        sequence_count += len(rows)
        matrices[split] = matrix(score_rows(model, tokenizer, rows, args.batch_size), labels)
    x_train, y_train = matrices["train"]
    x_test, y_test = matrices["test"]
    correct = y_test == 1
    payload = {
        "model_path": str(args.model_path),
        "cache_dir": str(args.cache_dir),
        "seed": args.seed,
        "samples": {"train": len(y_train), "test": len(y_test)},
        "teacher_forced_sequences": sequence_count,
        "cost_multiplier_vs_pre_only": 2.0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scores": {
            "pre_logp": score(y_test, x_test[:, 0]),
            "post_logp": score(y_test, x_test[:, 1]),
            "fixed_negative_amplification": score(y_test, x_test[:, 3]),
            "learned_pre_post_curve": learned_curve(x_train, y_train, x_test, y_test),
        },
        "amplification_summary": {
            "correct_mean": float(x_test[correct, 2].mean()),
            "incorrect_mean": float(x_test[~correct, 2].mean()),
            "incorrect_minus_correct": float(x_test[~correct, 2].mean() - x_test[correct, 2].mean()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
