"""Cue-ensemble lever + probe robustness (P1-4), on multi-cue confmc caches.

For each cut we have 3 semantically-equivalent cues, each giving (ans, first-token lp).
Tests, per cell (5-seed OOF AUROC on ChainUQ-style features built from confmc):
  single_cue0/1/2 : trajectory features from each cue alone (robustness: mean/worst)
  cue_mean        : per-cut confidence = mean lp over cues (denoised trajectory)
  cue_ensemble    : cue_mean features + cross-cue AGREEMENT features
                    (per-cut: #distinct answers across cues, lp spread) -> does cue
                    disagreement add signal?
Reports macro over the subset + Δ(ensemble - best single cue). Robustness = worst-cue
vs mean-cue gap. Requires labels + cdyn-style construction inline (no math_verify).
"""
from __future__ import annotations

import argparse, json, warnings
import numpy as np
warnings.filterwarnings("ignore")

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
    if x.ndim == 1: x = x.reshape(-1, 1)
    if not np.isfinite(x).all():
        cmin = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0)
        cmin = np.where(np.isfinite(cmin), cmin, 0.0)
        idx = np.where(~np.isfinite(x)); x[idx] = np.take(cmin, idx[1])
    return x


def oof(X, y):
    X = clean(X); acc = np.zeros(len(y))
    for s in SEEDS:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            acc[te] += c.predict_proba(X[te])[:, 1]
    return acc / len(SEEDS)


def traj_feats(ans_seq, lp_seq):
    """CONV(identity) + CDYN(confidence) features, matching the main pipeline."""
    neu = ans_seq; lp = lp_seq; n = len(neu); final = neu[-1] if neu else None
    half = n // 2
    agree = [1.0 if (a is not None and final is not None and _eq(a, final)) else 0.0 for a in neu]
    af = np.mean(agree) if n else 0.0; lh = np.mean(agree[half:]) if n - half > 0 else af
    run = 0
    for i in range(n-1, -1, -1):
        if agree[i] == 1.0: run += 1
        else: break
    fst = run/n if n else 0.0
    conv = 1.0
    for i in range(n):
        if all(agree[j] == 1.0 for j in range(i, n)): conv = (i+1)/n; break
    ids, reps = [], []
    for a in neu:
        if a is None: ids.append(-1); continue
        f = next((k for k, r in enumerate(reps) if _eq(a, r)), None)
        if f is None: reps.append(a); f = len(reps)-1
        ids.append(f)
    flip = sum(1 for i in range(1, n) if ids[i] != ids[i-1]) / max(1, n-1)
    CONV = [af, lh, fst, conv, flip, len(set(i for i in ids if i != -1))]
    lpv = [v for v in lp if v is not None]
    slope = float(np.polyfit(np.arange(len(lpv)), lpv, 1)[0]) if len(lpv) >= 2 else 0.0
    fc = -10.0
    for a, v in zip(neu, lp):
        if a is not None and final is not None and _eq(a, final) and v is not None: fc = v; break
    CDYN = [np.mean(lpv) if lpv else -10, lpv[-1] if lpv else -10, np.min(lpv) if lpv else -10,
            slope, fc, np.std(lpv) if len(lpv) > 1 else 0.0]
    return CONV + CDYN


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default=""); args = ap.parse_args()
    res = {"single0": {}, "single1": {}, "single2": {}, "cue_mean": {}, "cue_ensemble": {}}
    for tag in args.tags:
        mc = EXP_ROOT/"conf_mc"/f"{tag}_confmc.json"; lf = EXP_ROOT/"labels"/f"{tag}.json"
        if not (mc.exists() and lf.exists()): continue
        recs = json.loads(mc.read_text()); labs = json.loads(lf.read_text())
        F = {k: [] for k in res}; y = []
        for r in recs:
            if not r["intermediate"] or r["id"] not in labs: continue
            # per cue: ans/lp sequences
            cue_ans = {0: [], 1: [], 2: []}; cue_lp = {0: [], 1: [], 2: []}
            mean_ans, mean_lp, agree_feats = [], [], []
            for cut in r["intermediate"]:
                cs = {c["cue_id"]: c for c in cut["cues"]}
                for ci in (0, 1, 2):
                    cue_ans[ci].append(cs.get(ci, {}).get("ans"))
                    cue_lp[ci].append(cs.get(ci, {}).get("lp"))
                lps = [cs[ci]["lp"] for ci in (0,1,2) if ci in cs and cs[ci]["lp"] is not None]
                anss = [cs[ci]["ans"] for ci in (0,1,2) if ci in cs and cs[ci]["ans"] is not None]
                mean_lp.append(np.mean(lps) if lps else None)
                # majority answer across cues at this cut
                mean_ans.append(cs.get(0, {}).get("ans"))
                # cross-cue disagreement: distinct answers, lp spread
                nd = len(set(anss)) if anss else 3
                sp = (np.max(lps) - np.min(lps)) if len(lps) > 1 else 0.0
                agree_feats.append((nd, sp))
            F["single0"].append(traj_feats(cue_ans[0], cue_lp[0]))
            F["single1"].append(traj_feats(cue_ans[1], cue_lp[1]))
            F["single2"].append(traj_feats(cue_ans[2], cue_lp[2]))
            F["cue_mean"].append(traj_feats(mean_ans, mean_lp))
            nd_arr = [a[0] for a in agree_feats]; sp_arr = [a[1] for a in agree_feats]
            ens = traj_feats(mean_ans, mean_lp) + [np.mean(nd_arr), np.max(nd_arr),
                                                   np.mean(sp_arr), np.max(sp_arr)]
            F["cue_ensemble"].append(ens)
            y.append(labs[r["id"]])
        y = np.array(y)
        if len(y) < 30 or y.sum() == 0 or (1-y).sum() == 0: continue
        for k in res:
            res[k][tag] = roc_auc_score(y, oof(np.array(F[k]), y))

    def macro(k): v = list(res[k].values()); return np.mean(v) if v else float("nan"), len(v)
    print("\n=== CUE-ENSEMBLE lever + robustness (multi-cue subset) ===")
    for k in ["single0", "single1", "single2", "cue_mean", "cue_ensemble"]:
        mv, n = macro(k); print(f"  {k:14s} macro {mv:.3f} (n={n})")
    s = [macro(f"single{i}")[0] for i in range(3)]
    print(f"\n  cue robustness: mean-of-cues {np.mean(s):.3f}, worst-cue {min(s):.3f}, spread {max(s)-min(s):.3f}")
    bestsingle = max(s)
    print(f"  cue_mean - best_single_cue: {macro('cue_mean')[0]-bestsingle:+.3f}")
    print(f"  cue_ensemble - best_single_cue: {macro('cue_ensemble')[0]-bestsingle:+.3f}")
    print(f"  cue_ensemble - cue_mean: {macro('cue_ensemble')[0]-macro('cue_mean')[0]:+.3f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2); print("saved", args.out)


if __name__ == "__main__":
    main()
