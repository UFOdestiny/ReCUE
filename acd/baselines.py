"""Judge-free UQ signals computed from cached generations, plus AUROC eval.

Baselines (single generation):
  - maxprob / mean-logprob / perplexity  (sequence confidence)
  - mean token entropy (from top-k logprobs)
  - answer-token confidence (logprob around the boxed answer region)
Self-consistency (k samples, expensive):
  - agreement fraction of the majority answer
  - answer entropy over sampled answers

All signals return HIGH = more confident (higher => more likely correct),
so AUROC is computed directly against correctness labels.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from acd.env import extract_pred_math, verify_math
from acd import data as dv


# ----------------------------- correctness -----------------------------

def correctness(row: Dict[str, Any]) -> int:
    return dv.verify(row, row["primary_text"])


# ----------------------------- single-pass signals -----------------------------

def sig_mean_logprob(row) -> float:
    lp = row.get("chosen_logprobs") or []
    if not lp:
        return -1e9
    return float(np.mean(lp))


def sig_perplexity_conf(row) -> float:
    # negative perplexity so that higher = more confident
    lp = row.get("chosen_logprobs") or []
    if not lp:
        return -1e9
    return float(-math.exp(-np.mean(lp)))


def sig_min_logprob(row) -> float:
    lp = row.get("chosen_logprobs") or []
    if not lp:
        return -1e9
    return float(np.min(lp))


def sig_mean_entropy(row) -> float:
    # top-k entropy averaged; return NEGATIVE entropy so higher = more confident
    topk = row.get("topk_logprobs") or []
    if not topk:
        return 0.0
    ents = []
    for vals in topk:
        p = np.exp(np.array(vals))
        p = p / max(p.sum(), 1e-12)
        ents.append(-float(np.sum(p * np.log(p + 1e-12))))
    return float(-np.mean(ents))


def sig_answer_region_logprob(row, window: int = 30) -> float:
    # logprob of last `window` tokens (answer region confidence)
    lp = row.get("chosen_logprobs") or []
    if not lp:
        return -1e9
    return float(np.mean(lp[-window:]))


def _token_confidence(row):
    """DeepConf per-token confidence C_i = -mean of top-k token logprobs.

    Higher C_i => model more peaked/confident at token i. Uses cached top-k.
    """
    topk = row.get("topk_logprobs") or []
    conf = []
    for vals in topk:
        if not vals:
            continue
        conf.append(-float(np.mean(vals)))
    return conf


def sig_deepconf_bottom(row, group_frac: float = 0.1) -> float:
    """DeepConf (Fu et al. 2025) lowest-group-confidence for a single trace.

    Sliding window of the bottom group_frac fraction of tokens; trace score is
    the minimum group-mean confidence (the least-confident stretch). Single pass.
    """
    conf = _token_confidence(row)
    if not conf:
        return -1e9
    n = len(conf)
    w = max(1, int(round(group_frac * n)))
    c = np.asarray(conf)
    if w >= n:
        return float(c.mean())
    csum = np.cumsum(np.insert(c, 0, 0.0))
    means = (csum[w:] - csum[:-w]) / w
    return float(means.min())  # least-confident window; higher => more confident


def sig_deepconf_tail(row, tail_frac: float = 0.1) -> float:
    """DeepConf tail confidence: mean token confidence over the final tail_frac."""
    conf = _token_confidence(row)
    if not conf:
        return -1e9
    n = len(conf)
    t = max(1, int(round(tail_frac * n)))
    return float(np.mean(conf[-t:]))


def sig_self_certainty(row) -> float:
    """Self-certainty (Kang et al. 2025): mean_i KL(U || p_i), approx from top-k.

    KL(U||p) = -log|V| - (1/|V|) sum log p_j; we use the top-k proxy
    sum_j p_j log p_j (neg entropy) + log k, averaged over tokens. Monotone in
    peakedness; higher => more certain. Single pass.
    """
    topk = row.get("topk_logprobs") or []
    if not topk:
        return 0.0
    vals_list = []
    for vals in topk:
        if not vals:
            continue
        p = np.exp(np.array(vals))
        p = p / max(p.sum(), 1e-12)
        # negative entropy (peakedness), + log k constant is monotone-irrelevant
        vals_list.append(float(np.sum(p * np.log(p + 1e-12))))
    if not vals_list:
        return 0.0
    return float(np.mean(vals_list))


# ----------------------------- self-consistency -----------------------------

def sampled_answers(row) -> List[Optional[str]]:
    return [extract_pred_math(t) for t in row.get("samples", [])]


def sig_selfconsistency(row) -> float:
    ans = [a for a in sampled_answers(row) if a is not None]
    if not ans:
        return 0.0
    c = Counter(ans)
    top, n = c.most_common(1)[0]
    return n / len(ans)


def sig_answer_entropy(row) -> float:
    ans = [a for a in sampled_answers(row) if a is not None]
    if not ans:
        return -1e9
    c = Counter(ans)
    total = sum(c.values())
    ent = -sum((v / total) * math.log(v / total) for v in c.values())
    return -ent  # higher = more confident


SINGLE_PASS = {
    "mean_logprob": sig_mean_logprob,
    "min_logprob": sig_min_logprob,
    "perplexity": sig_perplexity_conf,
    "mean_entropy": sig_mean_entropy,
    "answer_region_lp": sig_answer_region_logprob,
    "deepconf_bottom": sig_deepconf_bottom,
    "deepconf_tail": sig_deepconf_tail,
    "self_certainty": sig_self_certainty,
}
SELF_CONSISTENCY = {
    "self_consistency": sig_selfconsistency,
    "answer_entropy": sig_answer_entropy,
}


def evaluate(rows: List[Dict[str, Any]], signals: Dict[str, Any]) -> Dict[str, Any]:
    labels = np.array([correctness(r) for r in rows])
    acc = float(labels.mean())
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    out = {"acc": acc, "n": len(rows), "n_correct": n_pos, "n_wrong": n_neg, "signals": {}}
    if n_pos == 0 or n_neg == 0:
        out["note"] = "degenerate labels; AUROC undefined"
        return out
    for name, fn in signals.items():
        scores = np.array([fn(r) for r in rows], dtype=float)
        # guard against constant / nan
        finite = np.isfinite(scores)
        if finite.sum() < len(scores):
            scores = np.where(finite, scores, np.nanmin(scores[finite]) if finite.any() else 0.0)
        try:
            auroc = roc_auc_score(labels, scores)
            prauc = average_precision_score(labels, scores)
        except Exception as e:
            auroc, prauc = float("nan"), float("nan")
        out["signals"][name] = {"auroc": round(float(auroc), 4), "prauc": round(float(prauc), 4)}
    return out
