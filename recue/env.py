"""Shared utilities for judge-free UQ experiments.

Everything path-related is read from the repo-root .env (anonymized).
No judge model anywhere: correctness labels come from deterministic verifiers.
"""
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> Dict[str, str]:
    env_path = REPO_ROOT / ".env"
    env: Dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env


ENV = load_env()
MODELS_ROOT = os.environ.get("MODELS_ROOT", "")
HF_CACHE = os.environ.get("HF_CACHE", "")
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "")
EXP_ROOT = Path(os.environ.get("EXP_ROOT", str(REPO_ROOT / "exp_out")))
EXP_ROOT.mkdir(parents=True, exist_ok=True)


def model_path(name: str) -> str:
    p = Path(MODELS_ROOT) / name
    return str(p)


# ----------------------------- answer verification -----------------------------

def _last_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content, brace-balanced."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j]
    return None


def normalize_num(s: str) -> Optional[str]:
    """Normalize a numeric string for exact comparison."""
    if s is None:
        return None
    s = s.strip()
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\!", "").replace(" ", "")
    s = re.sub(r"\\text\{.*?\}", "", s)
    s = s.strip()
    # strip trailing period
    s = s.rstrip(".")
    # try float
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except Exception:
        pass
    return s if s else None


def extract_pred_math(text: str) -> Optional[str]:
    """Extract a predicted answer from a math generation.

    Priority: \\boxed{}, then 'answer is X', then last number.
    """
    b = _last_boxed(text)
    if b is not None:
        n = normalize_num(b)
        if n is not None:
            return n
    # 'answer is ...'
    m = re.findall(r"answer\s*(?:is|:)?\s*\$?(-?[\d,\.]+)", text, flags=re.IGNORECASE)
    if m:
        return normalize_num(m[-1])
    # #### style (gsm8k)
    m = re.findall(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return normalize_num(m[-1])
    # last number
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if m:
        return normalize_num(m[-1])
    return None


def gsm8k_gold(answer_field: str) -> Optional[str]:
    m = re.findall(r"####\s*(-?[\d,\.]+)", answer_field)
    if m:
        return normalize_num(m[-1])
    return normalize_num(answer_field)


def verify_math(pred: Optional[str], gold: Optional[str]) -> int:
    if pred is None or gold is None:
        return 0
    if pred == gold:
        return 1
    # numeric tolerance
    try:
        if abs(float(pred) - float(gold)) < 1e-6:
            return 1
    except Exception:
        pass
    return 0


# ----------------------------- io -----------------------------

def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
