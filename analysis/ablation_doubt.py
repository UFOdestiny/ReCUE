"""Ablation with bidirectional (confirm/deny) contrastive doubt features.

Feature groups from the bidir cache:
  B0  answer-convergence (prior art): agree/stable/conv/flip on neutral answers
  C   forced-answer confidence dynamics: neutral_lp trajectory
  E   CONTRASTIVE doubt swing (new, strongest hypothesis):
        - confirm_flip: neutral vs confirm disagreement
        - deny_flip:    neutral vs deny disagreement
        - swing:        confirm vs deny disagreement (bidirectional instability)
        - deny_hold:    fraction where deny cue FAILS to change the answer (robustness)
Report B0, B0+C, B0+E, B0+C+E per dataset (5-seed CV).
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
from acd.features import _eq

SEEDS = [2026, 7, 13, 42, 100]


def clean(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.isfinite(x).all():
        cmin = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0)
        cmin = np.where(np.isfinite(cmin), cmin, 0.0)
        idx = np.where(~np.isfinite(x))
        x[idx] = np.take(cmin, idx[1])
    return x


def cv(X, y):
    X = clean(X); acc = np.zeros(len(y))
    for s in SEEDS:
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            oof[te] = c.predict_proba(X[te])[:, 1]
        acc += oof
    return roc_auc_score(y, acc / len(SEEDS))


def feats(rec):
    inter = rec["intermediate"]
    neu = [x["neutral"] for x in inter]
    lp = [x.get("neutral_lp") for x in inter]
    con = [x.get("confirm") for x in inter]
    den = [x.get("deny") for x in inter]
    n = len(neu); final = neu[-1] if neu else None; half = n // 2

    agree = [1.0 if (a is not None and final is not None and _eq(a, final)) else 0.0 for a in neu]
    agree_frac = np.mean(agree) if n else 0.0
    last_half = np.mean(agree[half:]) if n - half > 0 else agree_frac
    run = 0
    for i in range(n - 1, -1, -1):
        if agree[i] == 1.0: run += 1
        else: break
    fstable = run / n if n else 0.0
    conv = 1.0
    for i in range(n):
        if all(agree[j] == 1.0 for j in range(i, n)): conv = (i + 1) / n; break
    ids = []; reps = []
    for a in neu:
        if a is None: ids.append(-1); continue
        f = next((k for k, r in enumerate(reps) if _eq(a, r)), None)
        if f is None: reps.append(a); f = len(reps) - 1
        ids.append(f)
    flip = sum(1 for i in range(1, n) if ids[i] != ids[i - 1]) / max(1, n - 1)
    B0 = [agree_frac, last_half, fstable, conv, flip]

    lpv = [v for v in lp if v is not None]
    slope = np.polyfit(np.arange(len(lpv)), lpv, 1)[0] if len(lpv) >= 2 else 0.0
    C = [np.mean(lpv) if lpv else -10, lpv[-1] if lpv else -10,
         np.min(lpv) if lpv else -10, slope]

    def dis(a, b):
        return 0.0 if (a is not None and b is not None and _eq(a, b)) else 1.0
    cflip = [dis(a, c) for a, c in zip(neu, con)]
    dflip = [dis(a, d) for a, d in zip(neu, den)]
    swing = [dis(c, d) for c, d in zip(con, den)]
    # deny_hold: deny cue fails to move answer (answer robust to denial)
    deny_hold = np.mean([1.0 - x for x in dflip]) if n else 0.0
    E = [np.mean(cflip) if n else 1, np.mean(dflip) if n else 1,
         np.mean(swing) if n else 1, deny_hold,
         np.mean(swing[half:]) if n - half > 0 else (np.mean(swing) if n else 1)]
    return B0, C, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    print(f"{'setting':22s}{'B0':>7s}{'B0+C':>8s}{'B0+E':>8s}{'B0+CE':>8s}{'ΔC':>7s}{'ΔE':>7s}")
    print("-" * 67)
    agg = {k: [] for k in "B0 BC BE BCE".split()}
    for tag in args.tags:
        bf = EXP_ROOT / "bidir" / f"{tag}_bidir.json"
        lf = EXP_ROOT / "labels" / f"{tag}.json"
        if not bf.exists() or not lf.exists():
            continue
        recs = json.loads(bf.read_text()); labs = json.loads(lf.read_text())
        y, B0, C, E = [], [], [], []
        for r in recs:
            if not r["intermediate"] or r["id"] not in labs:
                continue
            b, c, e = feats(r); B0.append(b); C.append(c); E.append(e); y.append(labs[r["id"]])
        y = np.array(y)
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        B0 = np.array(B0); C = np.array(C); E = np.array(E)
        a0 = cv(B0, y); ac = cv(np.hstack([B0, C]), y)
        ae = cv(np.hstack([B0, E]), y); ace = cv(np.hstack([B0, C, E]), y)
        d = "_".join(tag.split("_")[:2])
        print(f"{d:22s}{a0:7.3f}{ac:8.3f}{ae:8.3f}{ace:8.3f}{ac-a0:+7.3f}{ae-a0:+7.3f}")
        agg["B0"].append(a0); agg["BC"].append(ac); agg["BE"].append(ae); agg["BCE"].append(ace)
    if agg["B0"]:
        print("-" * 67)
        print(f"{'MEAN':22s}{np.mean(agg['B0']):7.3f}{np.mean(agg['BC']):8.3f}"
              f"{np.mean(agg['BE']):8.3f}{np.mean(agg['BCE']):8.3f}"
              f"{np.mean(agg['BC'])-np.mean(agg['B0']):+7.3f}{np.mean(agg['BE'])-np.mean(agg['B0']):+7.3f}")
    print("\nE = contrastive confirm/deny doubt swing (bidirectional). ΔE>0 => novel beyond convergence.")


if __name__ == "__main__":
    main()
