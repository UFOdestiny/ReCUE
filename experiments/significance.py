"""Multi-seed significance for the headline comparisons.

Bootstrap over examples (paired) to get CIs and p-values on AUROC differences,
plus multi-seed variance for the supervised (CV) estimators. Fully offline.

Headline tests per dataset:
  - stab+lp(ours,1x) vs best 1x SOTA           (Direction 1)
  - hybrid@2(ours,~2.1x) vs self_consistency@8 (Direction 2)
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

SEEDS = [2026, 7, 13, 42, 100, 1, 2, 3, 4, 5]


def clean(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        m = np.nanmin(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0
        x = np.where(np.isfinite(x), x, m)
    return x


def cv_multiseed(X, y):
    """Return per-seed AUROC list (each seed = one full 5-fold OOF)."""
    X = clean(X)
    aurocs = []
    for s in SEEDS:
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            oof[te] = c.predict_proba(X[te])[:, 1]
        aurocs.append(roc_auc_score(y, oof))
    return np.array(aurocs)


def boot_pvalue(y, s_a, s_b, n=2000, seed=0):
    """Paired bootstrap: P(AUROC_a <= AUROC_b) for H0 a>b. Returns (mean_diff, p)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y); s_a = clean(s_a); s_b = clean(s_b)
    idx_pos = np.where(y == 1)[0]; idx_neg = np.where(y == 0)[0]
    diffs = []
    base = roc_auc_score(y, s_a) - roc_auc_score(y, s_b)
    for _ in range(n):
        p = rng.choice(idx_pos, len(idx_pos), replace=True)
        q = rng.choice(idx_neg, len(idx_neg), replace=True)
        ii = np.concatenate([p, q])
        yy = y[ii]
        try:
            d = roc_auc_score(yy, s_a[ii]) - roc_auc_score(yy, s_b[ii])
        except Exception:
            continue
        diffs.append(d)
    diffs = np.array(diffs)
    p_le = float(np.mean(diffs <= 0))  # prob ours not better
    return base, p_le, np.percentile(diffs, [2.5, 97.5])


def load(tag):
    gen = json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{tag}.json").read_text())
    feats = json.loads((EXP_ROOT / "feats" / f"{tag}.json").read_text())
    sampans = json.loads((EXP_ROOT / "sampans" / f"{tag}.json").read_text())
    ptrue_p = EXP_ROOT / "ptrue" / f"{tag}_ptrue.json"
    ptrue = json.loads(ptrue_p.read_text()) if ptrue_p.exists() else None
    y, F, lp, dyn, pt, ans = [], [], [], [], [], []
    for g in gen:
        i = g["id"]
        if i not in labs or i not in feats:
            continue
        y.append(labs[i]); F.append(feats[i]["feat"]); dyn.append(feats[i]["dyn_scalar"])
        lp.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        pt.append(ptrue.get(i, 0.5) if ptrue else np.nan)
        ans.append(sampans.get(i, []))
    return (np.array(y), clean(np.array(F)), clean(np.array(lp)),
            clean(np.array(dyn)), np.array(pt), ans)


def vote(ans, k):
    a = [x for x in ans[:k] if x is not None]
    return Counter(a).most_common(1)[0][1] / len(a) if a else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    print(f"{'dataset':10s} {'D1: ours1x vs SC@8':>26s} {'D2: hybrid@2 vs SC@8':>28s}")
    print("-" * 68)
    d1_diffs, d2_diffs = [], []
    for tag in args.tags:
        try:
            y, F, lp, dyn, pt, ans = load(tag)
        except FileNotFoundError:
            continue
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        d = tag.split("_")[0]
        sc8 = np.array([vote(a, 8) for a in ans])
        # ours single-gen (mean over multiseed CV, then use the mean-OOF for boot)
        # for the paired boot we need one score vector: use seed-averaged OOF
        def oof_mean(X):
            acc = np.zeros(len(y))
            for s in SEEDS[:5]:
                o = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(clean(X), y):
                    c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(clean(X)[tr], y[tr])
                    o[te] = c.predict_proba(clean(X)[te])[:, 1]
                acc += o
            return acc / 5
        ours1x = oof_mean(np.hstack([F, lp]))
        sc2 = np.array([vote(a, 2) for a in ans]).reshape(-1, 1)
        hyb2 = oof_mean(np.hstack([F, lp, sc2]))
        b1, p1, ci1 = boot_pvalue(y, ours1x, sc8)
        b2, p2, ci2 = boot_pvalue(y, hyb2, sc8)
        d1_diffs.append(b1); d2_diffs.append(b2)
        print(f"{d:10s}  Δ={b1:+.3f} p={p1:.3f} CI[{ci1[0]:+.3f},{ci1[1]:+.3f}]"
              f"   Δ={b2:+.3f} p={p2:.3f} CI[{ci2[0]:+.3f},{ci2[1]:+.3f}]")
    print("-" * 68)
    print(f"mean Δ  D1(ours1x - SC@8)={np.mean(d1_diffs):+.3f}   D2(hybrid@2 - SC@8)={np.mean(d2_diffs):+.3f}")
    print("(p = bootstrap prob ours NOT better; p<0.05 => ours significantly better)")


if __name__ == "__main__":
    main()
