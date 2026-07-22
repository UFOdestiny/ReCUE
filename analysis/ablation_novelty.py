"""Decisive novelty ablation: what beyond answer-convergence?

Reviewer's core question: prior work (Answer-Convergence, CGR, Prefix-Consistency)
already uses (a) whether/when the intermediate answer converges, and (b) single-point
answer certainty. Do we have signal BEYOND that?

We build nested feature sets from the challenge-probe cache (neutral answer sequence,
neutral forced-answer logprob per cut, challenge answer per cut) and measure
incremental CV-AUROC:

  B0  answer-convergence ONLY (prior art): does the intermediate answer stabilize
      to the final one, when, and how consistently.
        {agree_frac, last_half_agree, final_stable_run, conv_frac, flip_rate}
  +C  forced-answer CONFIDENCE dynamics (novel): trajectory of neutral_lp
        {mean_lp, last_lp, lp_slope, lp_min, lp_at_first_commit}
  +D  DOUBT-robustness (novel): challenge vs neutral disagreement
        {chal_flip_rate, chal_last_half_flip, chal_first_agree_frac}

Report B0, B0+C, B0+D, B0+C+D per dataset (5-seed CV). If +C / +D add AUROC on
top of B0, that is the differentiating contribution vs answer-convergence.
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
        col_min = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0)
        col_min = np.where(np.isfinite(col_min), col_min, 0.0)
        inds = np.where(~np.isfinite(x))
        x[inds] = np.take(col_min, inds[1])
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
    return roc_auc_score(y, acc / len(SEEDS))


def feats_from_rec(rec):
    inter = rec["intermediate"]
    ans = [x["neutral"] for x in inter]
    lp = [x["neutral_lp"] for x in inter]
    chal = [x["challenge"] for x in inter]
    n = len(ans)
    final = ans[-1] if ans else None

    # ---- B0: answer-convergence (prior art) ----
    agree = [1.0 if (a is not None and final is not None and _eq(a, final)) else 0.0 for a in ans]
    half = n // 2
    agree_frac = np.mean(agree) if n else 0.0
    last_half_agree = np.mean(agree[half:]) if n - half > 0 else agree_frac
    # final stable run
    run = 0
    for i in range(n - 1, -1, -1):
        if agree[i] == 1.0:
            run += 1
        else:
            break
    final_stable_run = run / n if n else 0.0
    # convergence fraction
    conv = 1.0
    for i in range(n):
        if all(agree[j] == 1.0 for j in range(i, n)):
            conv = (i + 1) / n
            break
    # flips
    ids = []
    reps = []
    for a in ans:
        if a is None:
            ids.append(-1); continue
        f = next((k for k, r in enumerate(reps) if _eq(a, r)), None)
        if f is None:
            reps.append(a); f = len(reps) - 1
        ids.append(f)
    flip = sum(1 for i in range(1, n) if ids[i] != ids[i - 1]) / max(1, n - 1)
    B0 = [agree_frac, last_half_agree, final_stable_run, conv, flip]

    # ---- +C: forced-answer confidence dynamics (novel) ----
    lpv = [v for v in lp if v is not None]
    mean_lp = np.mean(lpv) if lpv else -10.0
    last_lp = lpv[-1] if lpv else -10.0
    min_lp = np.min(lpv) if lpv else -10.0
    # slope over cut index
    if len(lpv) >= 2:
        xs = np.arange(len(lpv))
        slope = np.polyfit(xs, lpv, 1)[0]
    else:
        slope = 0.0
    # lp at first commit to final answer
    fc_lp = -10.0
    for a, v in zip(ans, lp):
        if a is not None and final is not None and _eq(a, final) and v is not None:
            fc_lp = v; break
    C = [mean_lp, last_lp, min_lp, slope, fc_lp]

    # ---- +D: doubt robustness (novel) ----
    dflip = [0.0 if (c is not None and a is not None and _eq(a, c)) else 1.0
             for a, c in zip(ans, chal)]
    d_flip_rate = np.mean(dflip) if n else 1.0
    d_last_half = np.mean(dflip[half:]) if n - half > 0 else d_flip_rate
    # fraction of cuts where challenge agrees with final answer
    d_final_agree = np.mean([1.0 if (c is not None and final is not None and _eq(c, final)) else 0.0
                             for c in chal]) if n else 0.0
    D = [d_flip_rate, d_last_half, d_final_agree]

    return B0, C, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    print(f"{'setting':22s}{'B0(conv)':>10s}{'B0+C':>8s}{'B0+D':>8s}{'B0+C+D':>9s}"
          f"{'ΔC':>7s}{'ΔD':>7s}")
    print("-" * 71)
    agg = {k: [] for k in ["B0", "BC", "BD", "BCD"]}
    for tag in args.tags:
        cf = EXP_ROOT / "challenge" / f"{tag}_chal.json"
        lf = EXP_ROOT / "labels" / f"{tag}.json"
        if not cf.exists() or not lf.exists():
            continue
        recs = json.loads(cf.read_text())
        labs = json.loads(lf.read_text())
        y, B0, C, D = [], [], [], []
        for r in recs:
            if not r["intermediate"] or r["pid" if "pid" in r else "id"] not in labs:
                continue
            key = r.get("id") or r.get("pid")
            b, c, d = feats_from_rec(r)
            B0.append(b); C.append(c); D.append(d); y.append(labs[key])
        y = np.array(y)
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        B0 = np.array(B0); C = np.array(C); D = np.array(D)
        a_b0 = cv(B0, y)
        a_bc = cv(np.hstack([B0, C]), y)
        a_bd = cv(np.hstack([B0, D]), y)
        a_bcd = cv(np.hstack([B0, C, D]), y)
        d = "_".join(tag.split("_")[:2])
        print(f"{d:22s}{a_b0:10.3f}{a_bc:8.3f}{a_bd:8.3f}{a_bcd:9.3f}"
              f"{a_bc-a_b0:+7.3f}{a_bd-a_b0:+7.3f}")
        agg["B0"].append(a_b0); agg["BC"].append(a_bc); agg["BD"].append(a_bd); agg["BCD"].append(a_bcd)
    if agg["B0"]:
        print("-" * 71)
        print(f"{'MEAN':22s}{np.mean(agg['B0']):10.3f}{np.mean(agg['BC']):8.3f}"
              f"{np.mean(agg['BD']):8.3f}{np.mean(agg['BCD']):9.3f}"
              f"{np.mean(agg['BC'])-np.mean(agg['B0']):+7.3f}{np.mean(agg['BD'])-np.mean(agg['B0']):+7.3f}")
    print("\nB0 = answer-convergence (prior art). +C = forced-answer confidence dynamics.")
    print("+D = doubt-injection robustness. ΔC/ΔD>0 => novel signal beyond answer-convergence.")


if __name__ == "__main__":
    main()
