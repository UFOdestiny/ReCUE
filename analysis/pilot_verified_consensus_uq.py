#!/usr/bin/env python3
"""Same-model P(True) and verified-consensus pilot.

The generator verifies its own modal and alternative answers against the
original context with a strict Yes/No prompt.  No external judge or gold
support annotation is used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

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
from analysis.pilot_contrastive_answer_uq import bootstrap, context_text, score
from data.hotpotqa import HotpotQADataset
from utils.prompting import build_chat_prompt_input, prompt_to_token_ids


VERIFY_SYSTEM = (
    "You are a strict evidence verifier. Decide whether the proposed answer is fully supported "
    "by the provided context and answers the question. Reply with exactly Yes or No."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("popllm/models/Llama-3.1-8B-Instruct"))
    parser.add_argument("--dataset-path", type=Path, default=Path("popllm/datasets/hotpot_qa/hf_dataset"))
    parser.add_argument(
        "--contrastive-result",
        type=Path,
        default=Path("popllm/results/refactor/pilot_contrastive_answer_uq_llama.json"),
    )
    parser.add_argument(
        "--score-cache",
        type=Path,
        default=Path("popllm/results/refactor/verified_answer_scores_hotpot_llama.json"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_verified_consensus_uq_llama.json"),
    )
    return parser.parse_args()


def compute_verifier_scores(args, diagnostics, dataset):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows, candidate_metadata = [], []
    flat_index = 0
    for sample_index, (diagnostic, item) in enumerate(zip(diagnostics, dataset.data)):
        for candidate in diagnostic["candidates"]:
            prompt = build_chat_prompt_input(
                tokenizer,
                [
                    {"role": "system", "content": VERIFY_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context_text(item)}\n\nQuestion: {item['question']}\n\n"
                            f"Proposed answer: {candidate}"
                        ),
                    },
                ],
                add_generation_prompt=True,
            )
            prompt_ids = prompt_to_token_ids(tokenizer, prompt)
            for verdict in ("yes", "no"):
                target_ids = tokenizer(" " + verdict.capitalize(), add_special_tokens=False)["input_ids"]
                room = args.max_length - len(target_ids)
                input_ids = prompt_ids[-room:] + target_ids
                rows.append(
                    {
                        "example_idx": flat_index,
                        "variant": verdict,
                        "input_ids": input_ids,
                        "target_start": len(input_ids) - len(target_ids),
                    }
                )
            candidate_metadata.append(
                {"flat_index": flat_index, "sample_index": sample_index, "candidate": candidate}
            )
            flat_index += 1
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    raw_scores = score_rows(model, tokenizer, rows, args.batch_size)
    probabilities: Dict[int, Dict[str, float]] = {}
    for metadata in candidate_metadata:
        values = raw_scores[metadata["flat_index"]]
        yes, no = values["yes"], values["no"]
        maximum = max(yes, no)
        p_true = math.exp(yes - maximum) / (math.exp(yes - maximum) + math.exp(no - maximum))
        probabilities.setdefault(metadata["sample_index"], {})[metadata["candidate"]] = p_true
    payload = {str(index): values for index, values in probabilities.items()}
    args.score_cache.parent.mkdir(parents=True, exist_ok=True)
    args.score_cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return probabilities


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    contrastive = json.loads(args.contrastive_result.read_text(encoding="utf-8"))
    diagnostics = contrastive["diagnostics"]
    dataset = HotpotQADataset(
        split="test", max_samples=len(diagnostics), dataset_path=str(args.dataset_path)
    )
    if args.score_cache.exists():
        verifier_scores = {
            int(index): values
            for index, values in json.loads(args.score_cache.read_text(encoding="utf-8")).items()
        }
    else:
        verifier_scores = compute_verifier_scores(args, diagnostics, dataset)

    features, labels, output_diagnostics = [], [], []
    for index, diagnostic in enumerate(diagnostics):
        modal = diagnostic["modal_answer"]
        p_true = verifier_scores[index]
        if modal not in p_true:
            continue
        counts = Counter(diagnostic["answers"])
        self_consistency = counts[modal] / sum(counts.values())
        candidate_logps = diagnostic["candidate_logps"]
        names = list(candidate_logps)
        logps = np.asarray([candidate_logps[name] for name in names])
        likelihood = np.exp(logps - logps.max())
        contrastive_mass = float(likelihood[names.index(modal)] / likelihood.sum())
        direct_logp = candidate_logps[modal]
        modal_ptrue = p_true[modal]
        alternatives = [value for answer, value in p_true.items() if answer != modal]
        strongest_alternative = max(alternatives) if alternatives else 0.0
        verifier_margin = modal_ptrue - strongest_alternative
        verifier_mass = modal_ptrue / max(1e-8, sum(p_true.values()))
        verified_consensus = math.sqrt(self_consistency * modal_ptrue)
        full_verified = (self_consistency * contrastive_mass * modal_ptrue) ** (1 / 3)
        features.append(
            [self_consistency, direct_logp, contrastive_mass, modal_ptrue, verifier_margin,
             verifier_mass, verified_consensus, full_verified]
        )
        labels.append(diagnostic["correct"])
        output_diagnostics.append({**diagnostic, "p_true": p_true})
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    indices = np.arange(len(y))
    train_index, test_index = train_test_split(
        indices, test_size=0.5, random_state=args.seed, stratify=y
    )
    answer_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    ).fit(x[train_index, :3], y[train_index])
    verifier_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    ).fit(x[train_index], y[train_index])
    test_methods = {
        "self_consistency": x[test_index, 0],
        "detached_direct_likelihood": x[test_index, 1],
        "contrastive_likelihood_mass": x[test_index, 2],
        "modal_p_true": x[test_index, 3],
        "verifier_margin": x[test_index, 4],
        "verifier_mass": x[test_index, 5],
        "fixed_verified_consensus": x[test_index, 6],
        "fixed_full_verified_consensus": x[test_index, 7],
        "learned_answer_only_stack": answer_model.predict_proba(x[test_index, :3])[:, 1],
        "learned_verifier_stack": verifier_model.predict_proba(x[test_index])[:, 1],
    }
    full_methods = {
        "self_consistency": x[:, 0],
        "detached_direct_likelihood": x[:, 1],
        "contrastive_likelihood_mass": x[:, 2],
        "modal_p_true": x[:, 3],
        "verifier_margin": x[:, 4],
        "verifier_mass": x[:, 5],
        "fixed_verified_consensus": x[:, 6],
        "fixed_full_verified_consensus": x[:, 7],
    }
    strongest = full_methods["detached_direct_likelihood"]
    payload = {
        "experiment": "same-model verified consensus",
        "model_path": str(args.model_path),
        "seed": args.seed,
        "samples": len(y),
        "correct_modal_answers": int(y.sum()),
        "test_results": {name: score(y[test_index], values) for name, values in test_methods.items()},
        "full_fixed_results": {name: score(y, values) for name, values in full_methods.items()},
        "paired_bootstrap_delta_vs_detached_direct_likelihood": {
            name: bootstrap(y, values, strongest, args.bootstrap, args.seed)
            for name, values in full_methods.items() if name != "detached_direct_likelihood"
        },
        "mean_features_by_correctness": {
            str(label): x[y == label].mean(axis=0).tolist() for label in (0, 1)
        },
        "diagnostics": output_diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
