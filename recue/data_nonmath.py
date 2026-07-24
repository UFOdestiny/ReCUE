"""Non-math reasoning benchmarks with judge-free (exact-match) correctness.

All tasks are multiple-choice, so the gold answer is a single option LETTER and
verification is deterministic letter-match — no judge model, consistent with the
math pipeline. Used for P0-2 (non-math generalization).

Tasks:
  bbh_logical   BBH logical_deduction_five_objects   (5-way)
  bbh_tracking  BBH tracking_shuffled_objects_three  (3-way)
  bbh_date      BBH date_understanding               (6-way)
  gpqa          GPQA-diamond (ungated mirror)         (4-way, science reasoning)

Each row: {id, question (with lettered options), gold (LETTER), type:"mc",
           n_choices}. `question` already embeds the options so the model sees them.
"""
from __future__ import annotations

import os
import re
import string
from typing import List, Dict, Any

_ABC = string.ascii_uppercase


def _hf_load(name, config=None, split="test"):
    os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", ""))
    from datasets import load_dataset
    return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)


def _fmt_options(opts: List[str]) -> str:
    return "\n".join(f"({_ABC[i]}) {o}" for i, o in enumerate(opts))


def _bbh(task: str, prefix: str, limit: int = 0) -> List[Dict[str, Any]]:
    d = _hf_load("lukaemon/bbh", task, "test")
    rows = []
    for i, r in enumerate(d):
        q = r["input"]                # already contains "Options:\n(A) ...(B)..."
        tgt = r["target"].strip()     # e.g. "(B)"
        m = re.search(r"\(([A-Z])\)", tgt)
        gold = m.group(1) if m else tgt.strip("() ")
        # count options present in the prompt
        letters = re.findall(r"\(([A-Z])\)", q)
        nch = len(set(letters)) if letters else 0
        rows.append({"id": f"{prefix}-{i}", "question": q, "gold": gold,
                     "type": "mc", "n_choices": nch})
    return rows[:limit] if limit else rows


def load_bbh_logical(limit: int = 0):
    return _bbh("logical_deduction_five_objects", "bbh_logical", limit)


def load_bbh_tracking(limit: int = 0):
    return _bbh("tracking_shuffled_objects_three_objects", "bbh_tracking", limit)


def load_bbh_date(limit: int = 0):
    return _bbh("date_understanding", "bbh_date", limit)


def load_gpqa(limit: int = 0) -> List[Dict[str, Any]]:
    """fingertap/GPQA-diamond: `question` has embedded a)/b)/c)/d), `answer` is a
    letter. We renormalise the letters to uppercase (A)-(D) form for a uniform prompt."""
    d = _hf_load("fingertap/GPQA-diamond", None, "test")
    rows = []
    for i, r in enumerate(d):
        q = r["question"].strip()
        gold = str(r["answer"]).strip().upper()[:1]
        # normalise "a)"/"A." option markers to "(A)" so the forced-answer probe
        # and letter extraction are consistent with BBH
        q = re.sub(r"(?m)^\s*([A-Da-d])[\.\)]\s", lambda m: f"({m.group(1).upper()}) ", q)
        rows.append({"id": f"gpqa-{i}", "question": q, "gold": gold,
                     "type": "mc", "n_choices": 4})
    return rows[:limit] if limit else rows


LOADERS = {
    "bbh_logical": load_bbh_logical,
    "bbh_tracking": load_bbh_tracking,
    "bbh_date": load_bbh_date,
    "gpqa": load_gpqa,
}

MC_TASKS = set(LOADERS.keys())


def extract_choice(text: str, n_choices: int = 26) -> str | None:
    """Extract a single option letter from a generation. Priority: \\boxed{X},
    'answer is (X)'/'answer: X', a lone parenthesised '(X)', else first standalone
    capital letter within range."""
    valid = set(_ABC[:n_choices]) if n_choices else set(_ABC)
    # \boxed{X} or \boxed{(X)}
    m = re.search(r"\\boxed\{\s*\(?([A-Za-z])\)?\s*\}", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    # answer is (X) / answer: X
    m = re.findall(r"answer\s*(?:is|:)?\s*\(?([A-Za-z])\)?\b", text, flags=re.IGNORECASE)
    for c in reversed(m):
        if c.upper() in valid:
            return c.upper()
    # parenthesised letter
    m = re.findall(r"\(([A-Za-z])\)", text)
    for c in reversed(m):
        if c.upper() in valid:
            return c.upper()
    # bare letter
    m = re.findall(r"\b([A-Z])\b", text)
    for c in reversed(m):
        if c in valid:
            return c
    return None


def verify_mc(row: Dict[str, Any], generation_text: str) -> int:
    pred = extract_choice(generation_text, row.get("n_choices", 26))
    gold = str(row.get("gold", "")).strip().upper()[:1]
    return int(pred is not None and pred == gold)
