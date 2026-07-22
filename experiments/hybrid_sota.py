"""Explore designs that use answer-confidence dynamics to BEAT SC@8 (AUROC).

Candidates (per cell, 5-seed CV, bootstrap CI vs SC@8):
  SC@8                : 8-sample vote fraction                      (8x, the SOTA)
  ours+SC8_fuse       : ours(conv+cdyn+seqlp) + SC@8 stats, logreg  (~8x, strictly more info)
  dyn_weighted_vote   : SC vote but each sample weighted by our single-chain confidence
                        (higher-confidence samples count more) -- score = weighted margin
  sem_entropy@8       : semantic entropy over answer clusters       (8x)
Key contrast: does ours+SC8_fuse SIGNIFICANTLY beat SC@8? If yes across most cells, our
signal is complementary to self-consistency and improves the SOTA at matched sample budget.
"""
from __future__ import annotations

import argparse
import json
import math
import numpy as np
from collections import Counter, defaultdict

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


def weighted_vote(a, w, k):
    """vote fraction weighted by per-sample weight w (softmax of sample confidence)."""
    aa = a[:k]; ww = w[:k]
    pairs = [(x, wi) for x, wi in zip(aa, ww) if x is not None and wi is not None]
    if not pairs:
        return 0.0
    ws = np.array([p[1] for p in pairs]); ws = np.exp(ws - ws.max()); ws = ws / ws.sum()
    agg = defaultdict(float)
    for (x, _), wi in zip(pairs, ws):
        agg[x] += wi
    return max(agg.values())


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
    print(f"{'tag':24s}{'SC@8':>7s}{'semE8':>7s}{'dynWvote':>9s}{'ours+SC8':>9s}"
          f"{'Δfuse':>7s}{'CI95':>16s}{'p':>7s}")
    print("-" * 96)
    dd = []; sig = 0; rows = 0
    for tag in args.tags:
        cdf = EXP_ROOT / "cdyn" / f"{tag}.json"; lf = EXP_ROOT / "labels" / f"{tag}.json"
        gf = EXP_ROOT / "gen" / f"{tag}.json"; saf = EXP_ROOT / "sampans" / f"{tag}.json"
        if not all(p.exists() for p in [cdf, lf, gf, saf]): continue
        cd = json.loads(cdf.read_text()); labs = json.loads(lf.read_text())
        gen = {g["id"]: g for g in json.loads(gf.read_text())}; sa = json.loads(saf.read_text())
        y, CONV, CDYN, SEQ, ans, smeanlp = [], [], [], [], [], []
        for i, c in cd.items():
            if i not in labs or i not in sa: continue
            CONV.append(c["conv"]); CDYN.append(c["cdyn"])
            g = gen.get(i, {}); SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
            ans.append(sa[i]); smeanlp.append(g.get("sample_meanlp") or [])
            y.append(labs[i])
        y = np.array(y)
        if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0: continue
        base = np.hstack([clean(np.array(CONV)), clean(np.array(CDYN)), clean(np.array(SEQ))])
        sc8 = np.array([vote(a, 8) for a in ans])
        se8 = np.array([sem(a, 8) for a in ans])
        # dyn-weighted vote: weight samples by their own mean logprob (if available)
        has_w = any(len(w) > 0 for w in smeanlp)
        dynw = np.array([weighted_vote(a, w, 8) if has_w else vote(a, 8)
                         for a, w in zip(ans, smeanlp)])
        # fusion: ours features + SC@8 statistics (vote + sem-entropy)
        fuse = oof(np.hstack([base, sc8.reshape(-1, 1), se8.reshape(-1, 1)]), y)
        au = lambda s: roc_auc_score(y, s)
        md, lo, hi, p = boot(y, fuse, sc8)
        dd.append(md); sig += (p < 0.05); rows += 1
        star = "*" if p < 0.05 else " "
        print(f"{tag:24s}{au(sc8):7.3f}{au(se8):7.3f}{au(dynw):9.3f}{au(fuse):9.3f}"
              f"{md:+7.3f}  [{lo:+.3f},{hi:+.3f}]{p:6.3f}{star}")
    print("-" * 96)
    print(f"ours+SC8_fuse - SC@8: mean {np.mean(dd):+.4f}, significantly > SC@8 in {sig}/{rows} cells")


if __name__ == "__main__":
    main()
