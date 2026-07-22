"""Priority-1 ablation (review3): is the signal genuine TRAJECTORY dynamics, or
just the final-probe confidence + answer-convergence?

Feature groups (computed per cell from conf caches):
  FINAL  : final-probe answer-confidence only            [last_lp]
  CONV   : answer-convergence / identity dynamics         [agree,last_half,stable,conv,flip,ndist]
  TRAJ   : PURE confidence-trajectory SHAPE, excluding the final value
           [mean_lp_excl_last, slope, std, min, early_mean(first third),
            late-minus-early gap, area-under-confidence, n_rises]
Configs compared (5-seed CV AUROC, problem-level bootstrap CI vs the key contrast):
  FINAL
  CONV
  CONV+FINAL              (= "answer-convergence + final logprob" — the null hypothesis)
  CONV+FINAL+TRAJ  (full)
  CONV+TRAJ (no final)
Decisive: (CONV+FINAL+TRAJ) > (CONV+FINAL), and (CONV+TRAJ) > (CONV+FINAL),
=> trajectory shape carries signal beyond final confidence.
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


def oof(X, y):
    X = clean(X); acc = np.zeros(len(y))
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        acc += o
    return acc / len(SEEDS)


def feats(rec):
    inter = rec["intermediate"]; neu = [x["neutral"] for x in inter]
    lp = [x.get("neutral_lp") for x in inter]
    n = len(neu); final = neu[-1] if neu else None; half = n // 2
    # CONV
    agree = [1.0 if (a is not None and final is not None and _eq(a, final)) else 0.0 for a in neu]
    af = np.mean(agree) if n else 0.0; lh = np.mean(agree[half:]) if n - half > 0 else af
    run = 0
    for i in range(n - 1, -1, -1):
        if agree[i] == 1.0: run += 1
        else: break
    fst = run / n if n else 0.0
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
    CONV = [af, lh, fst, conv, flip, len(set(i for i in ids if i != -1))]
    # confidence values
    lpv = [v for v in lp if v is not None]
    FINAL = [lpv[-1] if lpv else -10.0]
    # TRAJ: exclude the final value
    body = lpv[:-1] if len(lpv) >= 2 else lpv
    if len(body) >= 1:
        mean_excl = np.mean(body)
        mn = np.min(body)
        third = max(1, len(lpv) // 3)
        early = np.mean(lpv[:third])
        late_gap = (np.mean(lpv[-third:]) - early)
        area = np.trapz(lpv) / max(len(lpv) - 1, 1)
        nrise = sum(1 for i in range(1, len(lpv)) if lpv[i] > lpv[i - 1]) / max(1, len(lpv) - 1)
        slope = np.polyfit(np.arange(len(lpv)), lpv, 1)[0] if len(lpv) >= 2 else 0.0
        std = np.std(lpv) if len(lpv) > 1 else 0.0
    else:
        mean_excl = mn = early = late_gap = area = -10.0; nrise = slope = std = 0.0
    TRAJ = [mean_excl, slope, std, mn, early, late_gap, area, nrise]
    return CONV, FINAL, TRAJ


def boot_p(y, sA, sB, n=1000, seed=0):
    rng = np.random.RandomState(seed); y = np.asarray(y); sA = np.asarray(sA); sB = np.asarray(sB)
    ip = np.where(y == 1)[0]; ineg = np.where(y == 0)[0]; d = []
    for _ in range(n):
        p = rng.choice(ip, len(ip), True); q = rng.choice(ineg, len(ineg), True)
        ii = np.concatenate([p, q]); yy = y[ii]
        try:
            d.append(roc_auc_score(yy, sA[ii]) - roc_auc_score(yy, sB[ii]))
        except Exception:
            pass
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d <= 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    agg = {k: [] for k in ["FINAL", "CONV", "CF", "FULL", "CT"]}
    d_traj_over_cf = []   # FULL - CF (does traj add over conv+final?)
    sig = 0
    results = []
    for tag in args.tags:
        cf = EXP_ROOT / "conf" / f"{tag}_conf.json"
        lf = EXP_ROOT / "labels" / f"{tag}.json"
        if not (cf.exists() and lf.exists()):
            continue
        recs = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
        y, CONV, FINAL, TRAJ = [], [], [], []
        for r in recs:
            if not r["intermediate"] or r["id"] not in labs:
                continue
            cv_, fn_, tr_ = feats(r); CONV.append(cv_); FINAL.append(fn_); TRAJ.append(tr_)
            y.append(labs[r["id"]])
        y = np.array(y)
        if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0:
            continue
        CONV = np.array(CONV); FINAL = np.array(FINAL); TRAJ = np.array(TRAJ)
        s_final = oof(FINAL, y); s_conv = oof(CONV, y)
        s_cf = oof(np.hstack([CONV, FINAL]), y)
        s_full = oof(np.hstack([CONV, FINAL, TRAJ]), y)
        s_ct = oof(np.hstack([CONV, TRAJ]), y)
        a = lambda s: roc_auc_score(y, s)
        mean_d, lo, hi, p = boot_p(y, s_full, s_cf, n=args.boot)
        d_traj_over_cf.append(mean_d); sig += (p < 0.05)
        results.append((tag, len(y), a(s_final), a(s_conv), a(s_cf), a(s_full), a(s_ct), mean_d, lo, hi, p))
        for k, v in zip(["FINAL", "CONV", "CF", "FULL", "CT"], [a(s_final), a(s_conv), a(s_cf), a(s_full), a(s_ct)]):
            agg[k].append(v)

    print(f"\n{'tag':24s}{'n':>5s}{'FINAL':>7s}{'CONV':>7s}{'C+F':>7s}{'FULL':>7s}{'C+TRAJ':>8s}"
          f"{'ΔTRAJ':>7s}{'CI95':>16s}{'p':>7s}")
    print("-" * 104)
    for (tag, n, af, ac, acf, afu, act, md, lo, hi, p) in results:
        star = "*" if p < 0.05 else " "
        print(f"{tag:24s}{n:5d}{af:7.3f}{ac:7.3f}{acf:7.3f}{afu:7.3f}{act:8.3f}"
              f"{md:+7.3f}  [{lo:+.3f},{hi:+.3f}]{p:6.3f}{star}")
    print("-" * 104)
    print(f"{'MACRO':24s}{'':>5s}{np.mean(agg['FINAL']):7.3f}{np.mean(agg['CONV']):7.3f}"
          f"{np.mean(agg['CF']):7.3f}{np.mean(agg['FULL']):7.3f}{np.mean(agg['CT']):8.3f}"
          f"{np.mean(d_traj_over_cf):+7.3f}")
    print(f"\nΔTRAJ = FULL(conv+final+traj) - CF(conv+final): mean {np.mean(d_traj_over_cf):+.4f}, "
          f"significant {sig}/{len(results)} cells")
    print(f"C+TRAJ (no final) macro {np.mean(agg['CT']):.3f} vs C+F macro {np.mean(agg['CF']):.3f} "
          f"(does trajectory-without-final beat conv+final?)")
    if args.out:
        json.dump({"results": [list(r) for r in results]}, open(args.out, "w"), indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
