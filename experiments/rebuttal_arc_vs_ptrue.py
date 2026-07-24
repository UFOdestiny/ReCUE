"""Decisive: is ARC just P(True)/self-verification? Matched supervision + head.

P(True) = model's self-evaluated probability its answer is correct (Kadavath 2022).
ARC     = re-elicit the answer at the completed prefix; encode agreement with the
          returned answer + full-answer likelihood + first-token confidence + convergence.

Both use the SAME correctness labels, SAME logistic head, SAME folds. If ARC and
P(True) were the same construct, (a) P(True) would add nothing to ARC and (b) ARC
would add nothing to P(True) and (c) the bare re-commitment AGREEMENT bit would be
redundant with P(True). We test all three on the 25 cells where P(True) parses.

Rows (macro AUROC over the matched 25-cell set):
  ptrue                         P(True) scalar -> head
  agree_only                    single re-commitment agreement bit -> head
  arc                           full ARC (agree+likelihood+conf+convergence)
  ptrue + agree_only            does the agreement bit add over P(True)?
  ptrue + arc                   does full ARC add over P(True)?
  arc + ptrue                   (same features; contrast is arc+ptrue - arc)
Contrasts (cell-equal hierarchical bootstrap):
  arc - ptrue, (ptrue+agree) - ptrue, (ptrue+arc) - ptrue, (arc+ptrue) - arc
"""
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from recue.env import EXP_ROOT

SEEDS = [2026, 7, 13, 42, 100]


def clean(x):
    x = np.asarray(x, float); x = x.reshape(len(x), -1)
    if not np.isfinite(x).all():
        cm = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
        i = np.where(~np.isfinite(x)); x[i] = np.take(cm, i[1])
    return x


def oof(X, y):
    X = clean(X); a = np.zeros(len(y))
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        a += o
    return a / len(SEEDS)


def hb(cell, n=3000, seed=0):
    rng = np.random.RandomState(seed); ds = []
    for _ in range(n):
        cs = rng.choice(len(cell), len(cell), True); d = []
        for ci in cs:
            y, a, b = cell[ci]
            ip = np.where(y == 1)[0]; ng = np.where(y == 0)[0]
            if not len(ip) or not len(ng): continue
            ii = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ng, len(ng), True)]); yy = y[ii]
            try: d.append(roc_auc_score(yy, a[ii]) - roc_auc_score(yy, b[ii]))
            except Exception: pass
        if d: ds.append(np.mean(d))
    ds = np.array(ds)
    return float(np.mean(ds)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(np.mean(ds <= 0))


Z = np.load(f"{EXP_ROOT}/ladder_feats.npz")
tags = sorted({k.split("::")[0] for k in Z.files})
g = lambda t, n: Z[f"{t}::{n}"]

scores = defaultdict(dict); ys = {}; used = []
corr_stats = []
for tag in tags:
    pt = g(tag, "PT").ravel()
    if not np.isfinite(pt).all():
        continue  # only cells where P(True) parses (matched set)
    y = g(tag, "y"); ys[tag] = y; used.append(tag)
    AGREE = g(tag, "AGREE"); ARC = np.hstack([g(tag, b) for b in ["AGREE", "FLL", "FCONF", "CONVP"]])
    PT = pt.reshape(-1, 1)
    scores["ptrue"][tag] = oof(PT, y)
    scores["agree_only"][tag] = oof(AGREE, y)
    scores["arc"][tag] = oof(ARC, y)
    scores["ptrue+agree"][tag] = oof(np.hstack([PT, AGREE]), y)
    scores["ptrue+arc"][tag] = oof(np.hstack([PT, ARC]), y)
    # raw correlation between P(True) and the re-commitment agreement bit
    a = AGREE.ravel()
    if a.std() > 0:
        corr_stats.append(np.corrcoef(pt, a)[0, 1])

ROWS = ["ptrue", "agree_only", "arc", "ptrue+agree", "ptrue+arc"]
print(f"matched cells (P(True) parses): {len(used)}")
print(f"{'row':16s}{'macroAUROC':>12s}")
for r in ROWS:
    print(f"{r:16s}{np.mean([roc_auc_score(ys[t], scores[r][t]) for t in used]):12.3f}")
print(f"\nmean |corr(P(True), agreement-bit)| = {np.mean(np.abs(corr_stats)):.3f}  "
      f"(low => they measure different things)")

print("\n=== contrasts (cell-equal hierarchical bootstrap, matched 25 cells) ===")
out = {}
for name, A, B in [("arc - ptrue", "arc", "ptrue"),
                   ("(ptrue+agree) - ptrue", "ptrue+agree", "ptrue"),
                   ("(ptrue+arc) - ptrue", "ptrue+arc", "ptrue"),
                   ("(ptrue+arc) - arc", "ptrue+arc", "arc")]:
    cell = [(ys[t], scores[A][t], scores[B][t]) for t in used]
    m, lo, hi, p = hb(cell)
    wins = sum(roc_auc_score(ys[t], scores[A][t]) > roc_auc_score(ys[t], scores[B][t]) for t in used)
    sig = "SIG" if lo > 0 or hi < 0 else "ns"
    print(f"  {name:24s} d{m:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} [{sig}] wins {wins}/{len(used)}")
    out[name] = [m, lo, hi, p, wins, len(used)]

json.dump({"n_cells": len(used),
           "macro_auroc": {r: float(np.mean([roc_auc_score(ys[t], scores[r][t]) for t in used])) for r in ROWS},
           "mean_abs_corr_ptrue_agree": float(np.mean(np.abs(corr_stats))),
           "contrasts": out}, open(f"{EXP_ROOT}/arc_vs_ptrue.json", "w"), indent=2)
print("saved arc_vs_ptrue.json")
