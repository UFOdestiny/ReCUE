#!/usr/bin/env python3
"""Evidence-intervention pilot for counterfactual hallucination detection.

For each cached HotpotQA response, this script teacher-forces its generated
conclusion under three contexts while holding its generated reasoning prefix
fixed: full evidence, gold supporting evidence only, and supporting evidence
deleted.  A small logistic probe tests whether the response curve contains
correctness information beyond ordinary answer likelihood.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    parser.add_argument(
        "--prefix-mode",
        choices=("generated", "none"),
        default="generated",
        help="Use the cached reasoning prefix or score the conclusion directly.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popllm/results/refactor/pilot_counterfactual_uq.json"),
    )
    return parser.parse_args()


def first_chunk(cache_dir: Path, split: str) -> list:
    paths = sorted(
        path for path in (cache_dir / split).glob("chunk_*.pt")
        if not path.name.endswith("_reasoning.pt")
    )
    if not paths:
        raise FileNotFoundError(cache_dir / split)
    return torch.load(paths[0], map_location="cpu", weights_only=False)


def clean_answer(sample: dict) -> str:
    claim = sample.get("conclusion_claim") or {}
    answer = str(claim.get("text", "")).split("```", 1)[0].strip().strip(" .")
    answer = re.sub(r"\s+", " ", answer)
    if not answer or len(answer.split()) > 20 or len(answer) > 160:
        return ""
    return answer


def reasoning_prefix(sample: dict) -> str:
    text = str(sample.get("generated_text", ""))
    match = re.search(r"(?im)^\s*conclusion\s*[:\-]", text)
    prefix = text[: match.start()] if match else ""
    prefix = prefix.strip()
    if not prefix:
        prefix = "Reasoning:\n"
    return prefix + "\nConclusion:"


def balanced_samples(cache: list, per_class: int, seed: int) -> list:
    groups = {0: [], 1: []}
    for sample in cache:
        label = sample.get("label")
        if label in groups and clean_answer(sample):
            groups[label].append(sample)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    take = min(per_class, len(groups[0]), len(groups[1]))
    selected = groups[0][:take] + groups[1][:take]
    rng.shuffle(selected)
    return selected


def context_variants(raw: dict) -> Dict[str, str]:
    titles = raw.get("context", {}).get("title", [])
    paragraphs = raw.get("context", {}).get("sentences", [])
    supporting = raw.get("supporting_facts", {})
    support_pairs = set(zip(supporting.get("title", []), supporting.get("sent_id", [])))
    full_blocks, support_blocks, deleted_blocks = [], [], []
    for title, sentences in zip(titles, paragraphs):
        full = [str(sentence) for sentence in sentences]
        kept_support = [sentence for idx, sentence in enumerate(full) if (title, idx) in support_pairs]
        kept_deleted = [sentence for idx, sentence in enumerate(full) if (title, idx) not in support_pairs]
        full_blocks.append(f"[{title}] {''.join(full)}")
        if kept_support:
            support_blocks.append(f"[{title}] {''.join(kept_support)}")
        if kept_deleted:
            deleted_blocks.append(f"[{title}] {''.join(kept_deleted)}")
    return {
        "full": "\n".join(full_blocks),
        "support_only": "\n".join(support_blocks),
        "support_deleted": "\n".join(deleted_blocks),
    }


def messages(system_prompt: str, question: str, context: str) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def make_sequences(
    tokenizer,
    dataset: HotpotQADataset,
    cache_samples: Sequence[dict],
    max_length: int,
    max_prefix_tokens: int,
    prefix_mode: str,
) -> Tuple[List[dict], List[int]]:
    rows, labels = [], []
    for example_idx, sample in enumerate(cache_samples):
        raw = dataset.data[int(sample["sample_id"])]
        answer = clean_answer(sample)
        prefix_text = reasoning_prefix(sample) if prefix_mode == "generated" else "Conclusion:"
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        prefix_ids = prefix_ids[-max_prefix_tokens:]
        target_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
        if not target_ids:
            continue
        variants = context_variants(raw)
        for variant, context in variants.items():
            prompt = build_chat_prompt_input(
                tokenizer,
                messages(dataset.system_prompt, raw["question"], context),
                add_generation_prompt=True,
            )
            prompt_ids = prompt_to_token_ids(tokenizer, prompt)
            room = max_length - len(prefix_ids) - len(target_ids)
            if room <= 32:
                continue
            prompt_ids = prompt_ids[-room:]
            rows.append(
                {
                    "example_idx": example_idx,
                    "variant": variant,
                    "input_ids": prompt_ids + prefix_ids + target_ids,
                    "target_start": len(prompt_ids) + len(prefix_ids),
                    "answer": answer,
                }
            )
        labels.append(int(sample["label"]))
    return rows, labels


@torch.inference_mode()
def score_rows(model, tokenizer, rows: List[dict], batch_size: int) -> Dict[int, Dict[str, float]]:
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    output: Dict[int, Dict[str, float]] = {}
    # Length sorting materially reduces padding cost while preserving identifiers.
    ordered = sorted(rows, key=lambda row: len(row["input_ids"]))
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for idx, row in enumerate(batch):
            ids = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
            input_ids[idx, : len(ids)] = ids
            attention[idx, : len(ids)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.float()
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        next_tokens = input_ids[:, 1:]
        token_logp = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
        for idx, row in enumerate(batch):
            # token at target_start is predicted at target_start - 1.
            begin = max(int(row["target_start"]) - 1, 0)
            end = len(row["input_ids"]) - 1
            score = float(token_logp[idx, begin:end].mean().item())
            output.setdefault(int(row["example_idx"]), {})[str(row["variant"])] = score
        del logits, log_probs, token_logp
    return output


def matrix(scores: Dict[int, Dict[str, float]], labels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    rows, targets = [], []
    for idx, label in enumerate(labels):
        curve = scores.get(idx, {})
        if not all(key in curve for key in ("full", "support_only", "support_deleted")):
            continue
        full, support, deleted = curve["full"], curve["support_only"], curve["support_deleted"]
        rows.append([full, support, deleted, full - deleted, support - deleted, full - support])
        targets.append(label)
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=int)


def fit_and_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, columns: Iterable[int]) -> Dict[str, float]:
    columns = list(columns)
    scaler = StandardScaler().fit(x_train[:, columns])
    train = scaler.transform(x_train[:, columns])
    test = scaler.transform(x_test[:, columns])
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=2026).fit(train, y_train)
    probability = model.predict_proba(test)[:, 1]
    return {
        "auroc": float(roc_auc_score(y_test, probability)),
        "auprc": float(average_precision_score(y_test, probability)),
    }


def summarize_curves(x: np.ndarray, y: np.ndarray) -> Dict[str, dict]:
    names = ("full_logp", "support_only_logp", "support_deleted_logp", "full_minus_deleted", "support_minus_deleted", "full_minus_support")
    result = {}
    for idx, name in enumerate(names):
        result[name] = {
            "correct_mean": float(x[y == 1, idx].mean()),
            "incorrect_mean": float(x[y == 0, idx].mean()),
            "mean_difference": float(x[y == 1, idx].mean() - x[y == 0, idx].mean()),
        }
    return result


def fixed_score_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, dict]:
    scores = {
        "raw_full_logp": x[:, 0],
        "full_minus_deleted": x[:, 3],
        "support_minus_deleted": x[:, 4],
        "negative_full_minus_support": -x[:, 5],
    }
    return {
        name: {
            "auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
        }
        for name, score in scores.items()
    }


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
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()

    selected = {
        "train": balanced_samples(first_chunk(args.cache_dir, "train"), args.train_per_class, args.seed),
        "test": balanced_samples(first_chunk(args.cache_dir, "test"), args.test_per_class, args.seed + 1),
    }
    raw_datasets = {
        "train": HotpotQADataset(split="train", max_samples=5000, dataset_path=str(args.dataset_path)),
        "test": HotpotQADataset(split="test", max_samples=0, dataset_path=str(args.dataset_path)),
    }
    matrices = {}
    sequence_count = 0
    for split in ("train", "test"):
        rows, labels = make_sequences(
            tokenizer,
            raw_datasets[split],
            selected[split],
            args.max_length,
            args.max_prefix_tokens,
            args.prefix_mode,
        )
        sequence_count += len(rows)
        scores = score_rows(model, tokenizer, rows, args.batch_size)
        matrices[split] = matrix(scores, labels)
    x_train, y_train = matrices["train"]
    x_test, y_test = matrices["test"]
    results = {
        "model_path": str(args.model_path),
        "cache_dir": str(args.cache_dir),
        "gpu": torch.cuda.get_device_name(0),
        "samples": {"train": int(len(y_train)), "test": int(len(y_test))},
        "teacher_forced_sequences": sequence_count,
        "cost_multiplier_vs_direct": 3.0,
        "prefix_mode": args.prefix_mode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "models": {
            "direct_full_logp": fit_and_score(x_train, y_train, x_test, y_test, [0]),
            "sensitivity_only": fit_and_score(x_train, y_train, x_test, y_test, [3, 4, 5]),
            "counterfactual_curve": fit_and_score(x_train, y_train, x_test, y_test, range(6)),
        },
        "fixed_test_scores": fixed_score_metrics(x_test, y_test),
        "test_curve_summary": summarize_curves(x_test, y_test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
