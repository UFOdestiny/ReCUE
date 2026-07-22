"""Comprehensive comparison for both research directions (offline, cached).

Method groups (all judge-free):
  SINGLE-GENERATION (cost ~1x):
    - mean_logprob, mean_entropy, self_certainty, deepconf_tail, deepconf_bottom
    - dyn_scalar(ours), stability+logprob(ours)   [Direction 1 target: beat SOTA]
  ONE EXTRA FORWARD:
    - p_true
  MULTI-SAMPLE baselines (cost k x):
    - self_consistency@k, semantic_entropy@k, CISC@k (conf-weighted vote margin),
      deepconf_vote@k (mean per-sample confidence)
  HYBRID (ours, Direction 2): stability(1gen) + few samples fused
    - hybrid@2, hybrid@4

Reports AUROC (5-seed CV for supervised) with a cost column, per dataset + AVG.
"""
from __future__ import annotations

import argparse
import json
import math
import numpy as np
from collections import Counter

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from acd.env import EXP_ROOT
from acd import baselines as S

SEEDS = [2026, 7, 13, 42, 100]


def clean(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        m = np.nanmin(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0
        x = np.where(np.isfinite(x), x, m)
    return x


def cv(X, y):
    X = clean(X)
    acc = np.zeros(len(y))
    for s in SEEDS:
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            oof[te] = c.predict_proba(X[te])[:, 1]
        acc += oof
    return acc / len(SEEDS)


def _vote_frac(ans, k):
    a = [x for x in ans[:k] if x is not None]
    return Counter(a).most_common(1)[0][1] / len(a) if a else 0.0


def _sem_entropy(ans, k):
    a = [x for x in ans[:k] if x is not None]
    if not a:
        return 0.0
    c = Counter(a); tot = sum(c.values())
    return sum((v / tot) * math.log(v / tot) for v in c.values())  # neg entropy (higher=confident)


def _cisc(ans, meanlp, k, temp=1.0):
    """Confidence-Informed Self-Consistency: confidence-weighted vote fraction.

    Weight each sample by softmax(mean_logprob/temp); score = weight mass on the
    (weighted) majority answer. Higher => more confident.
    """
    aa = ans[:k]; ww = meanlp[:k]
    pairs = [(a, w) for a, w in zip(aa, ww) if a is not None and w is not None]
    if not pairs:
        return 0.0
    ws = np.array([w for _, w in pairs]) / temp
    ws = np.exp(ws - ws.max()); ws = ws / ws.sum()
    agg = {}
    for (a, _), w in zip(pairs, ws):
        agg[a] = agg.get(a, 0.0) + w
    return max(agg.values())


def _deepconf_vote(meanlp, k):
    """Mean per-sample confidence over the k samples (DeepConf voting proxy)."""
    v = [w for w in meanlp[:k] if w is not None]
    return float(np.mean(v)) if v else -1e9


def load(tag, kmax=8):
    gen = json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{tag}.json").read_text())
    feats = json.loads((EXP_ROOT / "feats" / f"{tag}.json").read_text())
    sampans = json.loads((EXP_ROOT / "sampans" / f"{tag}.json").read_text())
    ptrue_p = EXP_ROOT / "ptrue" / f"{tag}_ptrue.json"
    ptrue = json.loads(ptrue_p.read_text()) if ptrue_p.exists() else None

    rows = {"y": [], "F": [], "lp": [], "sp": {m: [] for m in
            ["mean_logprob", "mean_entropy", "self_certainty", "deepconf_tail", "deepconf_bottom"]},
            "dyn": [], "pt": [], "ans": [], "meanlp": []}
    for g in gen:
        i = g["id"]
        if i not in labs or i not in feats:
            continue
        rows["y"].append(labs[i])
        rows["F"].append(feats[i]["feat"])
        rows["dyn"].append(feats[i]["dyn_scalar"])
        rows["lp"].append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        for m in rows["sp"]:
            rows["sp"][m].append(S.SINGLE_PASS[m](g))
        rows["pt"].append(ptrue.get(i, 0.5) if ptrue else np.nan)
        rows["ans"].append(sampans.get(i, []))
        rows["meanlp"].append(g.get("sample_meanlp") or [None] * kmax)
    rows["y"] = np.array(rows["y"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    results = {}  # method -> {dataset: auroc}
    cost = {}
    ds_list = []
    for tag in args.tags:
        try:
            R = load(tag)
        except FileNotFoundError:
            continue
        y = R["y"]
        if len(y) == 0 or y.sum() == 0 or (1 - y).sum() == 0:
            continue
        d = tag.split("_")[0]
        ds_list.append(d)
        F = clean(np.array(R["F"])); lp = clean(np.array(R["lp"]))
        has_samplelp = any(v is not None for row in R["meanlp"] for v in (row or []))

        def put(name, scores, c):
            results.setdefault(name, {})[d] = roc_auc_score(y, clean(scores))
            cost[name] = c

        # single-generation
        for m in R["sp"]:
            put(m, np.array(R["sp"][m]), "1x")
        put("dyn_scalar(ours)", np.array(R["dyn"]), "1x*")
        put("stab+lp(ours,D1)", cv(np.hstack([F, lp]), y), "1x*")
        if R["pt"] is not None and np.isfinite(R["pt"]).any():
            put("p_true", np.array(R["pt"]), "1fwd")
        # multi-sample baselines
        for k in [2, 4, 8]:
            put(f"self_consistency@{k}", np.array([_vote_frac(a, k) for a in R["ans"]]), f"{k}x")
            put(f"sem_entropy@{k}", np.array([_sem_entropy(a, k) for a in R["ans"]]), f"{k}x")
            if has_samplelp:
                put(f"CISC@{k}", np.array([_cisc(a, m, k) for a, m in zip(R["ans"], R["meanlp"])]), f"{k}x")
                put(f"deepconf_vote@{k}", np.array([_deepconf_vote(m, k) for m in R["meanlp"]]), f"{k}x")
        # hybrid (ours, D2): stability + few samples
        for k in [2, 4]:
            sc = np.array([_vote_frac(a, k) for a in R["ans"]]).reshape(-1, 1)
            se = np.array([_sem_entropy(a, k) for a in R["ans"]]).reshape(-1, 1)
            put(f"hybrid@{k}(ours,D2)", cv(np.hstack([F, lp, sc, se]), y), f"~{k+0.1:.1f}x")

    order = ["mean_logprob", "mean_entropy", "self_certainty", "deepconf_tail",
             "deepconf_bottom", "p_true", "dyn_scalar(ours)", "stab+lp(ours,D1)",
             "self_consistency@2", "sem_entropy@2", "CISC@2", "deepconf_vote@2",
             "hybrid@2(ours,D2)",
             "self_consistency@4", "sem_entropy@4", "CISC@4", "deepconf_vote@4",
             "hybrid@4(ours,D2)",
             "self_consistency@8", "sem_entropy@8", "CISC@8", "deepconf_vote@8"]
    print(f"\n=== AUROC ({', '.join(ds_list)}) ===")
    hdr = f"{'method':22s}{'cost':>7s}" + "".join(f"{d[:8]:>9s}" for d in ds_list) + f"{'AVG':>8s}"
    print(hdr); print("-" * len(hdr))
    for m in order:
        if m not in results:
            continue
        vals = [results[m].get(d, float('nan')) for d in ds_list]
        print(f"{m:22s}{cost.get(m,''):>7s}" + "".join(f"{v:9.3f}" for v in vals) + f"{np.nanmean(vals):8.3f}")


if __name__ == "__main__":
    main()
