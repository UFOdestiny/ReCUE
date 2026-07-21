#!/usr/bin/env python3
"""Pilot for hallucination UQ after a self-cited evidence bottleneck."""

from __future__ import annotations

import argparse
import json
import random
import re
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

from analysis.pilot_counterfactual_uq import balanced_samples, clean_answer, first_chunk, messages, score_rows
from data.hotpotqa import HotpotQADataset
from utils.prompting import build_chat_prompt_input, prompt_to_token_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("popllm/models/Llama-3.1-8B-Instruct"))
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("popllm/cached_features/5k/hotpot_qa/Llama-3.1-8B-Instruct"),
    )
    parser.add_argument("--dataset-path", type=Path, default=Path("popllm/datasets/hotpot_qa/hf_dataset"))
    parser.add_argument("--train-per-class", type=int, default=32)
    parser.add_argument("--test-per-class", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output", type=Path,
        default=Path("popllm/results/refactor/pilot_evidence_bottleneck_llama.json"),
    )
    return parser.parse_args()


def unique(items):
    seen, output = set(), []
    for item in items:
        key = str(item).casefold()
        if key not in seen:
            seen.add(key)
            output.append(str(item))
    return output


def cited_titles(sample: dict, raw: dict):
    text = str(sample.get("generated_text", ""))
    conclusion = re.search(r"(?im)^\s*conclusion\s*[:\-]", text)
    if conclusion:
        text = text[: conclusion.start()]
    known = [str(title) for title in raw.get("context", {}).get("title", [])]
    cited = []
    for title in known:
        if re.search(re.escape(title), text, flags=re.IGNORECASE):
            cited.append(title)
    bracketed = re.findall(r"(?:passage\s*)?\[([^\]]+)\]", text, flags=re.IGNORECASE)
    known_map = {title.casefold(): title for title in known}
    cited.extend(known_map[value.strip().casefold()] for value in bracketed if value.strip().casefold() in known_map)
    return unique(cited)


STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "was", "were", "are", "for", "into",
    "states", "passage", "step", "therefore", "which", "who", "what", "when", "where",
    "has", "have", "had", "not", "but", "its", "his", "her", "their", "about", "also",
}


def words(text: str):
    return {
        token for token in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(token) > 2 and token not in STOPWORDS
    }


def reasoning_aligned_titles(sample: dict, raw: dict):
    text = str(sample.get("generated_text", ""))
    conclusion = re.search(r"(?im)^\s*conclusion\s*[:\-]", text)
    if conclusion:
        text = text[: conclusion.start()]
    steps = re.findall(r"(?im)^\s*step\s*\d+\s*[:\-]\s*(.+)$", text)
    titles = [str(title) for title in raw.get("context", {}).get("title", [])]
    paragraphs = raw.get("context", {}).get("sentences", [])
    passage_words = {
        title: words(title + " " + "".join(str(sentence) for sentence in sentences))
        for title, sentences in zip(titles, paragraphs)
    }
    selected = []
    for step in steps:
        step_words = words(step)
        if not step_words:
            continue
        scored = []
        for title, tokens in passage_words.items():
            overlap = len(step_words & tokens)
            score = overlap / max((len(step_words) * max(len(tokens), 1)) ** 0.5, 1e-8)
            scored.append((score, overlap, title))
        best = max(scored, default=(0.0, 0, ""))
        if best[1] > 0:
            selected.append(best[2])
    # Explicit citations are high-precision anchors; lexical alignment recovers
    # uncited premises required by multi-hop answers.
    return unique(cited_titles(sample, raw) + selected)


def contexts(sample: dict, raw: dict, seed: int):
    titles = [str(title) for title in raw.get("context", {}).get("title", [])]
    paragraphs = raw.get("context", {}).get("sentences", [])
    title_to_block = {
        title: f"[{title}] {''.join(str(sentence) for sentence in sentences)}"
        for title, sentences in zip(titles, paragraphs)
    }
    cited = cited_titles(sample, raw)
    aligned = reasoning_aligned_titles(sample, raw)
    gold = unique(raw.get("supporting_facts", {}).get("title", []))
    rng = random.Random(seed + int(sample.get("sample_id", 0)))
    candidates = [title for title in unique(titles) if title not in cited]
    random_titles = rng.sample(candidates, min(len(cited), len(candidates))) if cited else []
    aligned_candidates = [title for title in unique(titles) if title not in aligned]
    random_aligned = (
        rng.sample(aligned_candidates, min(len(aligned), len(aligned_candidates))) if aligned else []
    )

    def render(selected):
        return "\n".join(title_to_block[title] for title in selected if title in title_to_block)

    return {
        "full": render(titles),
        "random": render(random_titles),
        "self_cited": render(cited),
        "random_aligned": render(random_aligned),
        "reasoning_aligned": render(aligned),
        "gold_titles": render(gold),
    }, cited, aligned, gold


def make_rows(tokenizer, dataset, samples, max_length: int, seed: int):
    rows, labels, metadata = [], [], []
    prefix_ids = tokenizer("Conclusion:", add_special_tokens=False)["input_ids"]
    for example_idx, sample in enumerate(samples):
        raw = dataset.data[int(sample["sample_id"])]
        answer = clean_answer(sample)
        target_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
        variants, cited, aligned, gold = contexts(sample, raw, seed)
        for variant, context in variants.items():
            prompt = build_chat_prompt_input(
                tokenizer,
                messages(dataset.system_prompt, raw["question"], context),
                add_generation_prompt=True,
            )
            prompt_ids = prompt_to_token_ids(tokenizer, prompt)
            room = max_length - len(prefix_ids) - len(target_ids)
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
        overlap = len(set(map(str.casefold, cited)) & set(map(str.casefold, gold)))
        metadata.append(
            {
                "n_cited": len(cited),
                "n_aligned": len(aligned),
                "gold_title_recall": overlap / max(len(gold), 1),
                "aligned_gold_title_recall": (
                    len(set(map(str.casefold, aligned)) & set(map(str.casefold, gold))) / max(len(gold), 1)
                ),
            }
        )
    return rows, labels, metadata


def matrix(scores, labels):
    variants = (
        "full", "random", "self_cited", "random_aligned", "reasoning_aligned", "gold_titles"
    )
    x, y = [], []
    for idx, label in enumerate(labels):
        row = scores.get(idx, {})
        if not all(variant in row for variant in variants):
            continue
        full, random_score, cited, random_aligned, aligned, gold = (row[variant] for variant in variants)
        x.append(
            [full, random_score, cited, random_aligned, aligned, gold,
             cited - full, aligned - full, gold - full]
        )
        y.append(label)
    return np.asarray(x), np.asarray(y, dtype=int)


def score(y, values):
    return {
        "auroc": float(roc_auc_score(y, values)),
        "auprc": float(average_precision_score(y, values)),
    }


def learned(x_train, y_train, x_test, y_test):
    columns = [0, 4, 7]
    scaler = StandardScaler().fit(x_train[:, columns])
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=2026).fit(
        scaler.transform(x_train[:, columns]), y_train
    )
    return score(y_test, model.predict_proba(scaler.transform(x_test[:, columns]))[:, 1])


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
    ).eval()
    selected = {
        "train": balanced_samples(first_chunk(args.cache_dir, "train"), args.train_per_class, args.seed),
        "test": balanced_samples(first_chunk(args.cache_dir, "test"), args.test_per_class, args.seed + 1),
    }
    datasets = {
        "train": HotpotQADataset(split="train", max_samples=5000, dataset_path=str(args.dataset_path)),
        "test": HotpotQADataset(split="test", max_samples=0, dataset_path=str(args.dataset_path)),
    }
    matrices, test_metadata, sequence_count = {}, None, 0
    for split in ("train", "test"):
        rows, labels, metadata = make_rows(tokenizer, datasets[split], selected[split], args.max_length, args.seed)
        sequence_count += len(rows)
        matrices[split] = matrix(score_rows(model, tokenizer, rows, args.batch_size), labels)
        if split == "test":
            test_metadata = metadata
    x_train, y_train = matrices["train"]
    x_test, y_test = matrices["test"]
    metadata = test_metadata or []
    payload = {
        "model_path": str(args.model_path),
        "seed": args.seed,
        "samples": {"train": len(y_train), "test": len(y_test)},
        "teacher_forced_sequences": sequence_count,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scores": {
            "full_logp": score(y_test, x_test[:, 0]),
            "random_same_size_logp": score(y_test, x_test[:, 1]),
            "self_cited_logp": score(y_test, x_test[:, 2]),
            "random_aligned_logp": score(y_test, x_test[:, 3]),
            "reasoning_aligned_logp": score(y_test, x_test[:, 4]),
            "gold_title_logp_upper_bound": score(y_test, x_test[:, 5]),
            "self_cited_minus_full": score(y_test, x_test[:, 6]),
            "reasoning_aligned_minus_full": score(y_test, x_test[:, 7]),
            "learned_full_aligned_curve": learned(x_train, y_train, x_test, y_test),
        },
        "citation_summary": {
            "mean_cited_titles": float(np.mean([row["n_cited"] for row in metadata])),
            "mean_aligned_titles": float(np.mean([row["n_aligned"] for row in metadata])),
            "zero_citation_rate": float(np.mean([row["n_cited"] == 0 for row in metadata])),
            "mean_gold_title_recall": float(np.mean([row["gold_title_recall"] for row in metadata])),
            "mean_aligned_gold_title_recall": float(
                np.mean([row["aligned_gold_title_recall"] for row in metadata])
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
