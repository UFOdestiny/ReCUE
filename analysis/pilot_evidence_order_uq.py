#!/usr/bin/env python3
"""Evidence-order invariance UQ at matched sampling cost.

For each question, create K deterministic permutations of the context passages
and sample one trajectory from each.  The score is the fraction of permutations
that reproduce the modal answer from K ordinary same-prompt samples.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.pilot_contrastive_answer_uq import bootstrap, score
from analysis.pilot_multitrajectory_uq import correct, entropy, extract_answer
from data.hotpotqa import HotpotQADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("popllm/models/Llama-3.1-8B-Instruct"))
    parser.add_argument("--dataset-path", type=Path, default=Path("popllm/datasets/hotpot_qa/hf_dataset"))
    parser.add_argument(
        "--ordinary-cache",
        type=Path,
        default=Path("popllm/results/refactor/multitrajectory_hotpot_llama_generations.json"),
    )
    parser.add_argument(
        "--permutation-cache",
        type=Path,
        default=Path("popllm/results/refactor/evidence_order_hotpot_llama_generations.json"),
    )
    parser.add_argument("--permutations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_evidence_order_uq_llama.json"),
    )
    return parser.parse_args()


def generate_permutations(args, ordinary_rows, dataset):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    prompts, mapping = [], []
    for sample_index, item in enumerate(dataset.data):
        blocks = [
            f"[{title}] {''.join(sentences)}"
            for title, sentences in zip(
                item.get("context", {}).get("title", []),
                item.get("context", {}).get("sentences", []),
            )
        ]
        for permutation_index in range(args.permutations):
            order = list(range(len(blocks)))
            rng = random.Random(args.seed + 10007 * sample_index + permutation_index)
            rng.shuffle(order)
            context = "\n".join(blocks[index] for index in order)
            messages = [
                {"role": "system", "content": dataset.system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {item['question']}",
                },
            ]
            prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            mapping.append((sample_index, permutation_index, order))

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
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    requests = llm.generate(prompts, sampling, use_tqdm=True)
    output = [
        {
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "permutations": [None] * args.permutations,
        }
        for row in ordinary_rows
    ]
    for (sample_index, permutation_index, order), request in zip(mapping, requests):
        output[sample_index]["permutations"][permutation_index] = {
            "order": order,
            "text": request.outputs[0].text,
        }
    args.permutation_cache.parent.mkdir(parents=True, exist_ok=True)
    args.permutation_cache.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    ordinary = json.loads(args.ordinary_cache.read_text(encoding="utf-8"))
    dataset = HotpotQADataset(
        split="test", max_samples=len(ordinary), dataset_path=str(args.dataset_path)
    )
    if args.permutation_cache.exists():
        permuted = json.loads(args.permutation_cache.read_text(encoding="utf-8"))
    else:
        permuted = generate_permutations(args, ordinary, dataset)
    if len(permuted) != len(ordinary):
        raise ValueError("Permutation cache size mismatch")

    features, labels, diagnostics = [], [], []
    for ordinary_row, permutation_row in zip(ordinary, permuted):
        ordinary_answers = [extract_answer(value["text"]) for value in ordinary_row["trajectories"]]
        ordinary_counts = Counter(ordinary_answers)
        modal, modal_count = ordinary_counts.most_common(1)[0]
        ordinary_consistency = modal_count / len(ordinary_answers)
        ordinary_entropy = entropy(ordinary_counts.values())

        permutation_answers = [
            extract_answer(value["text"]) for value in permutation_row["permutations"]
        ]
        permutation_counts = Counter(permutation_answers)
        order_support = permutation_counts[modal] / len(permutation_answers)
        permutation_entropy = entropy(permutation_counts.values())
        combined = math.sqrt(ordinary_consistency * order_support)
        label = correct(modal, ordinary_row["ground_truth"])
        features.append(
            [ordinary_consistency, -ordinary_entropy, order_support, -permutation_entropy, combined]
        )
        labels.append(label)
        diagnostics.append(
            {
                "question": ordinary_row["question"],
                "ground_truth": ordinary_row["ground_truth"],
                "modal_answer": modal,
                "correct": label,
                "ordinary_consistency": ordinary_consistency,
                "order_support": order_support,
                "permutation_answers": permutation_answers,
            }
        )
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    indices = np.arange(len(y))
    train_index, test_index = train_test_split(
        indices, test_size=0.5, random_state=args.seed, stratify=y
    )
    baseline_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    ).fit(x[train_index, :2], y[train_index])
    invariance_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    ).fit(x[train_index, :4], y[train_index])
    test_methods = {
        "ordinary_self_consistency": x[test_index, 0],
        "negative_ordinary_entropy": x[test_index, 1],
        "evidence_order_support": x[test_index, 2],
        "negative_permutation_entropy": x[test_index, 3],
        "fixed_consistency_invariance_joint": x[test_index, 4],
        "learned_ordinary_stack": baseline_model.predict_proba(x[test_index, :2])[:, 1],
        "learned_invariance_stack": invariance_model.predict_proba(x[test_index, :4])[:, 1],
    }
    full_methods = {
        "ordinary_self_consistency": x[:, 0],
        "evidence_order_support": x[:, 2],
        "negative_permutation_entropy": x[:, 3],
        "fixed_consistency_invariance_joint": x[:, 4],
    }
    baseline = full_methods["ordinary_self_consistency"]
    payload = {
        "experiment": "evidence-order invariance UQ",
        "model_path": str(args.model_path),
        "seed": args.seed,
        "samples": len(y),
        "ordinary_generations_per_sample": len(ordinary[0]["trajectories"]),
        "permutation_generations_per_sample": args.permutations,
        "correct_original_modal_answers": int(y.sum()),
        "test_results": {name: score(y[test_index], values) for name, values in test_methods.items()},
        "full_fixed_results": {name: score(y, values) for name, values in full_methods.items()},
        "paired_bootstrap_delta_vs_ordinary_self_consistency": {
            name: bootstrap(y, values, baseline, args.bootstrap, args.seed)
            for name, values in full_methods.items() if name != "ordinary_self_consistency"
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
