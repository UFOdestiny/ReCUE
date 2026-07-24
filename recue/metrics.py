"""Calibration and selective-prediction metrics for UQ confidence scores.

All take conf in [0,1] (higher=more confident) and binary correctness labels.
"""
from __future__ import annotations
import numpy as np


def ece(conf, labels, n_bins=15):
    conf = np.asarray(conf, float); labels = np.asarray(labels, float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    N = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc = labels[m].mean()
        avg_conf = conf[m].mean()
        e += (m.sum() / N) * abs(acc - avg_conf)
    return float(e)


def brier(conf, labels):
    conf = np.asarray(conf, float); labels = np.asarray(labels, float)
    return float(np.mean((conf - labels) ** 2))


def aurc(conf, labels):
    """Area under risk-coverage curve (lower is better).

    Rank by confidence desc; risk = error rate over the most-confident coverage.
    """
    conf = np.asarray(conf, float); labels = np.asarray(labels, float)
    order = np.argsort(-conf)
    err = 1.0 - labels[order]
    cum_err = np.cumsum(err) / (np.arange(len(err)) + 1)
    return float(np.mean(cum_err))


def risk_at_coverage(conf, labels, coverage=0.5):
    conf = np.asarray(conf, float); labels = np.asarray(labels, float)
    order = np.argsort(-conf)
    k = max(1, int(round(coverage * len(conf))))
    sel = labels[order][:k]
    return float(1.0 - sel.mean())
