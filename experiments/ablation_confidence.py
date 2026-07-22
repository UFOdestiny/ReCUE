"""Comprehensive, reviewer-grade ablation across the full model x dataset matrix.

Uses conf caches (per-cut neutral answer + forced-answer logprob).

Feature groups:
  CONV : answer-convergence (prior art) - identity-based stabilization
         {agree_frac, last_half_agree, final_stable_run, conv_frac, flip_rate, n_distinct}
  CDYN : answer-CONFIDENCE dynamics (ours, novel) - trajectory of forced-answer logprob
         {mean_lp, last_lp, min_lp, lp_slope, lp_at_first_commit, lp_std}
  SEQLP: single-pass sequence logprob (baseline) {seq_mean_lp, seq_mean_ent}

Reports, per model x dataset AND macro/per-family, with problem-level bootstrap 95% CI:
  CONV | CONV+CDYN (ΔCDYN) | full | CDYN-alone
Establishes that confidence-dynamics adds signal BEYOND answer-convergence.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from collections import defaultdict

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
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.isfinite(x).all():
        cmin = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0)
        cmin = np.where(np.isfinite(cmin), cmin, 0.0)
        idx = np.where(~np.isfinite(x))
        x[idx] = np.take(cmin, idx[1])
    return x


def oof(X, y):
    X = clean(X)
    acc = np.zeros(len(y))
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        acc += o
    return acc / len(SEEDS)


def conv_cdyn(rec):
    inter = rec["intermediate"]
    neu = [x["neutral"] for x in inter]
    lp = [x.get("neutral_lp") for x in inter]
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
    ndist = len(set(i for i in ids if i != -1))
    CONV = [agree_frac, last_half, fstable, conv, flip, ndist]

    lpv = [v for v in lp if v is not None]
    slope = np.polyfit(np.arange(len(lpv)), lpv, 1)[0] if len(lpv) >= 2 else 0.0
    fc = -10.0
    for a, v in zip(neu, lp):
        if a is not None and final is not None and _eq(a, final) and v is not None:
            fc = v; break
    CDYN = [np.mean(lpv) if lpv else -10, lpv[-1] if lpv else -10,
            np.min(lpv) if lpv else -10, slope, fc, np.std(lpv) if len(lpv) > 1 else 0.0]
    return CONV, CDYN


def boot_ci(y, sA, sB, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); sA = np.asarray(sA); sB = np.asarray(sB)
    ip = np.where(y == 1)[0]; ineg = np.where(y == 0)[0]
    diffs = []
    for _ in range(n):
        p = rng.choice(ip, len(ip), True); q = rng.choice(ineg, len(ineg), True)
        ii = np.concatenate([p, q]); yy = y[ii]
        try:
            diffs.append(roc_auc_score(yy, sA[ii]) - roc_auc_score(yy, sB[ii]))
        except Exception:
            pass
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(np.mean(diffs <= 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    fam_delta = defaultdict(list)
    for tag in args.tags:
        cf = EXP_ROOT / "conf" / f"{tag}_conf.json"
        lf = EXP_ROOT / "labels" / f"{tag}.json"
        gf = EXP_ROOT / "gen" / f"{tag}.json"
        if not (cf.exists() and lf.exists()):
            continue
        recs = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
        gen = {g["id"]: g for g in json.loads(gf.read_text())}
        y, CONV, CDYN, SEQ = [], [], [], []
        for r in recs:
            if not r["intermediate"] or r["id"] not in labs:
                continue
            cv_, cd_ = conv_cdyn(r)
            CONV.append(cv_); CDYN.append(cd_)
            g = gen.get(r["id"], {})
            SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
            y.append(labs[r["id"]])
        y = np.array(y)
        if len(y) < 20 or y.sum() == 0 or (1 - y).sum() == 0:
            continue
        CONV = np.array(CONV); CDYN = np.array(CDYN); SEQ = clean(np.array(SEQ))
        s_conv = oof(CONV, y)
        s_full = oof(np.hstack([CONV, CDYN, SEQ]), y)
        s_convcd = oof(np.hstack([CONV, CDYN]), y)
        s_cdyn = oof(CDYN, y)
        a_conv = roc_auc_score(y, s_conv)
        a_convcd = roc_auc_score(y, s_convcd)
        a_full = roc_auc_score(y, s_full)
        a_cdyn = roc_auc_score(y, s_cdyn)
        lo, hi, p = boot_ci(y, s_convcd, s_conv, n=args.boot)
        fam = "_".join(tag.split("_")[1:-1])
        fam_delta[fam].append(a_convcd - a_conv)
        rows.append((tag, len(y), a_conv, a_cdyn, a_convcd, a_full, a_convcd - a_conv, lo, hi, p))

    print(f"\n{'tag':24s}{'n':>5s}{'CONV':>7s}{'CDYNa':>7s}{'C+CD':>7s}{'FULL':>7s}"
          f"{'ΔCD':>7s}{'CI95':>16s}{'p':>7s}")
    print("-" * 96)
    deltas, sig = [], 0
    for (tag, n, ac, acd, acc, af, d, lo, hi, p) in rows:
        star = "*" if p < 0.05 else " "
        print(f"{tag:24s}{n:5d}{ac:7.3f}{acd:7.3f}{acc:7.3f}{af:7.3f}{d:+7.3f}"
              f"  [{lo:+.3f},{hi:+.3f}]{p:6.3f}{star}")
        deltas.append(d); sig += (p < 0.05)
    print("-" * 96)
    print(f"{'MACRO MEAN':24s}{'':>5s}{np.mean([r[2] for r in rows]):7.3f}"
          f"{np.mean([r[3] for r in rows]):7.3f}{np.mean([r[4] for r in rows]):7.3f}"
          f"{np.mean([r[5] for r in rows]):7.3f}{np.mean(deltas):+7.3f}")
    print(f"\nΔCD (confidence-dynamics gain over answer-convergence): mean {np.mean(deltas):+.3f}, "
          f"significant {sig}/{len(rows)} cells")
    print("per-family mean ΔCD:")
    for fam, ds in sorted(fam_delta.items()):
        print(f"  {fam:16s} {np.mean(ds):+.3f}  (n={len(ds)})")
    if args.out:
        json.dump({"rows": [list(r) for r in rows],
                   "fam_delta": {k: v for k, v in fam_delta.items()}},
                  open(args.out, "w"), indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
