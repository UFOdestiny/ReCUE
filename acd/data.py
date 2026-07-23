"""Verifiable-answer datasets. Correctness = deterministic checker, no judge model."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from acd.env import DATASETS_ROOT, gsm8k_gold, extract_pred_math, verify_math, normalize_num

HUB = os.path.expanduser(DATASETS_ROOT) if DATASETS_ROOT else os.path.expanduser("~/.cache/huggingface/hub")


def _snap(name: str) -> str:
    cands = glob.glob(os.path.join(HUB, name, "snapshots", "*"))
    if not cands:
        raise FileNotFoundError(name)
    return cands[0]


def load_gsm8k(split: str = "test", limit: int = 0) -> List[Dict[str, Any]]:
    import pandas as pd
    p = glob.glob(os.path.join(_snap("datasets--openai--gsm8k"), "main", f"{split}-*.parquet"))[0]
    df = pd.read_parquet(p)
    rows = []
    for i, r in df.iterrows():
        rows.append({"id": f"gsm8k-{split}-{i}", "question": r["question"],
                     "gold": gsm8k_gold(r["answer"]), "type": "math"})
    if limit:
        rows = rows[:limit]
    return rows


def load_math500(limit: int = 0) -> List[Dict[str, Any]]:
    p = os.path.join(_snap("datasets--HuggingFaceH4--MATH-500"), "test.jsonl")
    rows = []
    for i, line in enumerate(open(p)):
        r = json.loads(line)
        rows.append({"id": f"math500-{i}", "question": r["problem"],
                     "gold": normalize_num(r["answer"]) or r["answer"].strip(),
                     "gold_raw": r["answer"].strip(), "type": "math",
                     "level": r.get("level")})
    if limit:
        rows = rows[:limit]
    return rows


def load_amc23(limit: int = 0) -> List[Dict[str, Any]]:
    import pandas as pd
    p = glob.glob(os.path.join(_snap("datasets--math-ai--amc23"), "test-*.parquet"))[0]
    df = pd.read_parquet(p)
    rows = []
    for i, r in df.iterrows():
        rows.append({"id": f"amc23-{i}", "question": r["question"],
                     "gold": normalize_num(str(r["answer"])), "type": "math"})
    if limit:
        rows = rows[:limit]
    return rows


def load_minerva(limit: int = 0) -> List[Dict[str, Any]]:
    p = os.path.join(_snap("datasets--math-ai--minervamath"), "test.jsonl")
    rows = []
    for i, line in enumerate(open(p)):
        r = json.loads(line)
        a = str(r.get("answer"))
        rows.append({"id": f"minerva-{i}", "question": r["question"],
                     "gold": normalize_num(a) or a.strip(), "gold_raw": a.strip(),
                     "type": "math"})
    if limit:
        rows = rows[:limit]
    return rows


def _to_scalar(a):
    import numpy as _np
    if isinstance(a, (list, tuple, _np.ndarray)):
        return a[0] if len(a) else None
    return a


def load_olympiad(limit: int = 0) -> List[Dict[str, Any]]:
    import pandas as pd
    p = os.path.join(_snap("datasets--math-ai--olympiadbench"), "test.parquet")
    df = pd.read_parquet(p)
    # text-only, single-answer, English subset for clean verifiable eval
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        if r.get("image_1") is not None:
            continue
        if bool(r.get("is_multiple_answer")):
            continue
        a = _to_scalar(r.get("final_answer"))
        if a is None:
            continue
        a = str(a)
        rows.append({"id": f"olympiad-{i}", "question": r["question"],
                     "gold": normalize_num(a) or a.strip(), "gold_raw": a.strip(),
                     "type": "math"})
    if limit:
        rows = rows[:limit]
    return rows


LOADERS = {
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "amc23": load_amc23,
    "minerva": load_minerva,
    "olympiad": load_olympiad,
}


def load_dataset(name: str, limit: int = 0) -> List[Dict[str, Any]]:
    name = name.lower()
    # non-math multiple-choice reasoning tasks (P0-2 generalization)
    try:
        from acd import data_nonmath as dnm
        if name in dnm.MC_TASKS:
            return dnm.LOADERS[name](limit=limit)
    except Exception:
        pass
    if name == "gsm8k":
        return load_gsm8k(limit=limit)
    return LOADERS[name](limit=limit)


_MV = None


def _get_mv():
    global _MV
    if _MV is None:
        try:
            from math_verify import parse as _p, verify as _v
            _MV = (_p, _v)
        except Exception:
            _MV = False
    return _MV


def verify(row: Dict[str, Any], generation_text: str) -> int:
    """Deterministic verification. Multiple-choice tasks -> exact letter match;
    math tasks -> math_verify (robust) with regex fallback. No judge model."""
    if row.get("type") == "mc":
        from acd import data_nonmath as dnm
        return dnm.verify_mc(row, generation_text)
    gold_raw = row.get("gold_raw") or row.get("gold")
    mv = _get_mv()
    if mv and gold_raw:
        _p, _v = mv
        try:
            gold_parsed = _p("$" + str(gold_raw) + "$")
            pred_parsed = _p(generation_text)
            if _v(gold_parsed, pred_parsed):
                return 1
        except Exception:
            pass
    # fallback: regex numeric match
    pred = extract_pred_math(generation_text)
    return verify_math(pred, row.get("gold"))
