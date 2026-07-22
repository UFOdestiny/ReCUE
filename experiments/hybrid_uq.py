"""Hybrid UQ: does fusing our single-chain signal with a FEW self-consistency
samples give a better CONFIDENCE score (AUROC/AURC) than expensive SC@8?

This is a UQ-quality claim (ranking correct vs wrong responses), NOT an
accuracy-allocation claim. Methods:
  ours(1x)           : conv+cdyn+seqlp        (single generation)
  SC@8               : 8-sample vote fraction (8x)
  hybrid@2 / @4      : ours + {sc,sem-entropy}@k fused by logreg (~2x / 4x)
Report AUROC with problem-level bootstrap CI for hybrid@k vs SC@8.
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
from acd import metrics as UQ

SEEDS = [2026, 7, 13, 42, 100]


def clean(x):
    x = np.asarray(x, float)
    if x.ndim == 1: x = x.reshape(-1, 1)
    if not np.isfinite(x).all():
        cmin = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0)
        cmin = np.where(np.isfinite(cmin), cmin, 0.0)
        idx = np.where(~np.isfinite(x)); x[idx] = np.take(cmin, idx[1])
    return x


def oof(X, y):
    X = clean(X); acc = np.zeros(len(y))
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        acc += o
    return acc / len(SEEDS)


def vote(a, k):
    aa = [x for x in a[:k] if x is not None]
    return Counter(aa).most_common(1)[0][1] / len(aa) if aa else 0.0


def sem(a, k):
    aa = [x for x in a[:k] if x is not None]
    if not aa: return 0.0
    c = Counter(aa); t = sum(c.values())
    return sum((v/t) * math.log(v/t) for v in c.values())


def boot(y, sA, sB, n=1000, seed=0):
    rng = np.random.RandomState(seed); y = np.asarray(y); sA = np.asarray(sA); sB = np.asarray(sB)
    ip = np.where(y == 1)[0]; ineg = np.where(y == 0)[0]; d = []
    for _ in range(n):
        p = rng.choice(ip, len(ip), True); q = rng.choice(ineg, len(ineg), True)
        ii = np.concatenate([p, q]); yy = y[ii]
        try: d.append(roc_auc_score(yy, sA[ii]) - roc_auc_score(yy, sB[ii]))
        except Exception: pass
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d <= 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    print(f"{'tag':24s}{'ours1x':>8s}{'SC@8':>7s}{'hyb@2':>7s}{'hyb@4':>7s}"
          f"{'h4-SC8':>8s}{'CI95':>16s}{'p':>7s}")
    print("-" * 92)
    d48 = []; sig = 0; rows = 0
    for tag in args.tags:
        cdf = EXP_ROOT / "cdyn" / f"{tag}.json"; lf = EXP_ROOT / "labels" / f"{tag}.json"
        gf = EXP_ROOT / "gen" / f"{tag}.json"; saf = EXP_ROOT / "sampans" / f"{tag}.json"
        if not all(p.exists() for p in [cdf, lf, gf, saf]): continue
        cd = json.loads(cdf.read_text()); labs = json.loads(lf.read_text())
        gen = {g["id"]: g for g in json.loads(gf.read_text())}; sa = json.loads(saf.read_text())
        y, CONV, CDYN, SEQ, ans = [], [], [], [], []
        for i, c in cd.items():
            if i not in labs or i not in sa: continue
            CONV.append(c["conv"]); CDYN.append(c["cdyn"])
            g = gen.get(i, {}); SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
            ans.append(sa[i]); y.append(labs[i])
        y = np.array(y)
        if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0: continue
        base = np.hstack([clean(np.array(CONV)), clean(np.array(CDYN)), clean(np.array(SEQ))])
        s_ours = oof(base, y)
        sc8 = np.array([vote(a, 8) for a in ans])
        sc2 = np.array([[vote(a, 2), sem(a, 2)] for a in ans])
        sc4 = np.array([[vote(a, 4), sem(a, 4)] for a in ans])
        s_h2 = oof(np.hstack([base, clean(sc2)]), y)
        s_h4 = oof(np.hstack([base, clean(sc4)]), y)
        au = lambda s: roc_auc_score(y, s)
        md, lo, hi, p = boot(y, s_h4, sc8)
        d48.append(md); sig += (p < 0.05); rows += 1
        star = "*" if p < 0.05 else " "
        print(f"{tag:24s}{au(s_ours):8.3f}{au(sc8):7.3f}{au(s_h2):7.3f}{au(s_h4):7.3f}"
              f"{md:+8.3f}  [{lo:+.3f},{hi:+.3f}]{p:6.3f}{star}")
    print("-" * 92)
    print(f"hyb@4 - SC@8: mean {np.mean(d48):+.4f}, hyb@4 significantly > SC@8 in {sig}/{rows} cells")


if __name__ == "__main__":
    main()
