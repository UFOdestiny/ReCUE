"""Evaluate ours vs Prefix-Consistency (and CoE if available) as per-response
correctness predictors (AUROC), with bootstrap CI for ours - baseline.

Prefix Consistency score (faithful to Iwase et al.): fraction of resampled
continuation answers (from the deepest / all prefixes) that match the original
final answer, via math_verify. We report:
  pc_deep   : reproduce fraction at the deepest prefix (closest to tau=0.75 spirit)
  pc_all    : mean reproduce fraction across all prefixes (smoother)
ours = cached OOF prob from conv+cdyn+seqlp.
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from acd.env import EXP_ROOT
from acd import baselines as S
from acd.features import _eq

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


def pc_scores(rec):
    final = rec["final_answer"]
    per_prefix = []
    for pf in rec["prefixes"]:
        ans = pf["answers"]
        if not ans:
            per_prefix.append(0.0); continue
        per_prefix.append(np.mean([1.0 if (a is not None and final is not None and _eq(a, final)) else 0.0
                                   for a in ans]))
    if not per_prefix:
        return 0.0, 0.0
    return per_prefix[-1], float(np.mean(per_prefix))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    print(f"{'tag':24s}{'n':>5s}{'pc_deep':>8s}{'pc_all':>8s}{'ours':>7s}"
          f"{'Δ(ours-pc)':>11s}{'CI95':>16s}{'p':>7s}")
    print("-" * 90)
    dd = []; sig = 0; rows = 0
    for tag in args.tags:
        pf = EXP_ROOT / "prefix" / f"{tag}_prefix.json"
        cdf = EXP_ROOT / "cdyn" / f"{tag}.json"
        lf = EXP_ROOT / "labels" / f"{tag}.json"
        gf = EXP_ROOT / "gen" / f"{tag}.json"
        if not all(p.exists() for p in [pf, cdf, lf, gf]):
            print(f"{tag}: missing ({[p.name for p in [pf,cdf,lf,gf] if not p.exists()]})"); continue
        recs = json.loads(pf.read_text()); cd = json.loads(cdf.read_text())
        labs = json.loads(lf.read_text()); gen = {g["id"]: g for g in json.loads(gf.read_text())}
        y, pcd, pca, CONV, CDYN, SEQ = [], [], [], [], [], []
        for r in recs:
            i = r["id"]
            if i not in labs or i not in cd:
                continue
            d1, d2 = pc_scores(r); pcd.append(d1); pca.append(d2)
            CONV.append(cd[i]["conv"]); CDYN.append(cd[i]["cdyn"])
            g = gen.get(i, {}); SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
            y.append(labs[i])
        y = np.array(y)
        if len(y) < 20 or y.sum() == 0 or (1 - y).sum() == 0:
            continue
        ours = oof(np.hstack([clean(np.array(CONV)), clean(np.array(CDYN)), clean(np.array(SEQ))]), y)
        au = lambda s: roc_auc_score(y, clean(np.asarray(s)).ravel())
        best_pc = np.array(pca) if au(pca) >= au(pcd) else np.array(pcd)
        md, lo, hi, p = boot(y, ours, best_pc)
        dd.append(md); sig += (p < 0.05); rows += 1
        star = "*" if p < 0.05 else " "
        print(f"{tag:24s}{len(y):5d}{au(pcd):8.3f}{au(pca):8.3f}{au(ours):7.3f}"
              f"{md:+11.3f}  [{lo:+.3f},{hi:+.3f}]{p:6.3f}{star}")
    if rows:
        print("-" * 90)
        print(f"ours - PrefixConsistency: mean {np.mean(dd):+.4f}, ours significantly > PC in {sig}/{rows} cells")


if __name__ == "__main__":
    main()
