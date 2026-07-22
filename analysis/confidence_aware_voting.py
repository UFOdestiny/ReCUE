"""Confidence-dynamics-aware self-consistency: beat SC@k using per-sample
confidence trajectories at MATCHED k-sample budget.

Per question, for each of k samples we have its answer + confidence trajectory
(forced-answer logprobs at cuts). We build cluster-level and question-level
features that self-consistency (vote fraction) ignores:
  vote_frac          : vanilla SC (baseline)
  conf_wtd_vote      : vote weighted by per-sample mean trajectory confidence
  conf_wtd_margin    : weighted top1-top2 margin
  best_cluster_conf  : mean confidence of the winning cluster's samples
  conf_gap           : winning-cluster conf - losing-cluster conf
  final_conf_top     : mean FINAL-cut confidence of winning-cluster samples
  traj_slope_top     : mean confidence slope of winning-cluster samples
  agree_conf_corr    : do high-confidence samples agree more? (corr of conf with modal-match)
Compare (5-seed CV, bootstrap CI vs vanilla SC@8):
  SC@8 (vote)  |  conf-aware (logreg over the features above)
"""
from __future__ import annotations

import argparse
import json
import math
import numpy as np
from collections import defaultdict, Counter

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
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        acc += o
    return acc / len(SEEDS)


def samp_traj(s):
    lps = [x["lp"] for x in s.get("inter", []) if x["lp"] is not None]
    if not lps:
        return {"mean": -10.0, "final": -10.0, "slope": 0.0, "min": -10.0}
    slope = np.polyfit(np.arange(len(lps)), lps, 1)[0] if len(lps) >= 2 else 0.0
    return {"mean": float(np.mean(lps)), "final": float(lps[-1]), "slope": float(slope),
            "min": float(np.min(lps))}


def cluster(ans_idx):
    """cluster answers by equivalence -> list of cluster member-index lists."""
    reps = []; clusters = []
    for i, a in ans_idx:
        placed = False
        for ci, rep in enumerate(reps):
            if _eq(a, rep):
                clusters[ci].append(i); placed = True; break
        if not placed:
            reps.append(a); clusters.append([i])
    return clusters


def feats(samps, k=None):
    if k is not None:
        samps = samps[:k]
    ans = [(i, s["final"]) for i, s in enumerate(samps) if s["final"] is not None]
    n = len(samps)
    if not ans:
        return [0, 0, 0, -10, 0, -10, 0, 0]
    trajs = [samp_traj(s) for s in samps]
    confs = np.array([trajs[i]["mean"] for i, _ in ans])
    w = np.exp(confs - confs.max()); w = w / w.sum()
    clusters = cluster(ans)
    # cluster weights (confidence-weighted mass) and plain counts
    cl_mass = []; cl_cnt = []; cl_finalconf = []; cl_slope = []
    idx_map = {i: k for k, (i, _) in enumerate(ans)}
    for cl in clusters:
        members = [idx_map[i] for i in cl]
        cl_mass.append(sum(w[m] for m in members))
        cl_cnt.append(len(cl))
        cl_finalconf.append(np.mean([trajs[i]["final"] for i in cl]))
        cl_slope.append(np.mean([trajs[i]["slope"] for i in cl]))
    order = np.argsort(-np.array(cl_mass))
    top = order[0]
    vote_frac = max(cl_cnt) / len(ans)
    conf_wtd_vote = cl_mass[top]
    top2_mass = cl_mass[order[1]] if len(order) > 1 else 0.0
    conf_wtd_margin = cl_mass[top] - top2_mass
    best_cluster_conf = cl_finalconf[top]
    conf_gap = cl_finalconf[top] - (np.mean([cl_finalconf[o] for o in order[1:]]) if len(order) > 1 else -10)
    final_conf_top = cl_finalconf[top]
    traj_slope_top = cl_slope[top]
    # do more-confident samples agree with the modal (plain) answer?
    modal = clusters[int(np.argmax(cl_cnt))]
    modal_set = set(modal)
    match = np.array([1.0 if ai in modal_set else 0.0 for ai, _ in ans])
    corr = float(np.corrcoef(confs, match)[0, 1]) if len(set(match)) > 1 else 0.0
    return [vote_frac, conf_wtd_vote, conf_wtd_margin, best_cluster_conf, conf_gap,
            final_conf_top, traj_slope_top, corr]


def vote_only(samps, k=None):
    if k is not None:
        samps = samps[:k]
    ans = [s["final"] for s in samps if s["final"] is not None]
    if not ans: return 0.0
    cl = cluster([(i, a) for i, a in enumerate(ans)])
    return max(len(c) for c in cl) / len(ans)


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
    ap.add_argument("--frontier", action="store_true",
                    help="compare conf-aware@k vs SC@k for k in {2,4,8}")
    args = ap.parse_args()
    au = lambda y, s: roc_auc_score(y, clean(np.asarray(s)).ravel())

    if args.frontier:
        # KEY TEST: does conf-aware@4 beat vanilla SC@8?
        print(f"{'tag':22s}{'n':>4s}"
              f"{'SC@2':>7s}{'CA@2':>7s}{'SC@4':>7s}{'CA@4':>7s}{'SC@8':>7s}{'CA@8':>7s}"
              f"{'CA4-SC8':>9s}{'p':>6s}")
        print("-" * 88)
        d48, sig48, rows = [], 0, 0
        for tag in args.tags:
            scf = EXP_ROOT / "sampleconf" / f"{tag}_sc.json"; lf = EXP_ROOT / "labels" / f"{tag}.json"
            if not (scf.exists() and lf.exists()):
                continue
            recs = json.loads(scf.read_text()); labs = json.loads(lf.read_text())
            rr = [r for r in recs if r["id"] in labs and r["samples"]]
            y = np.array([labs[r["id"]] for r in rr])
            if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0:
                continue
            out = {}
            for k in [2, 4, 8]:
                scv = np.array([vote_only(r["samples"], k) for r in rr])
                ca = oof(clean(np.array([feats(r["samples"], k) for r in rr])), y)
                out[f"sc{k}"] = au(y, scv); out[f"ca{k}"] = au(y, ca)
                if k == 8: sc8 = scv
                if k == 4: ca4 = ca
            md, lo, hi, p = boot(y, ca4, sc8)
            d48.append(md); sig48 += (p < 0.05 and md > 0); rows += 1
            star = "*" if (p < 0.05 and md > 0) else " "
            print(f"{tag:22s}{len(y):4d}"
                  f"{out['sc2']:7.3f}{out['ca2']:7.3f}{out['sc4']:7.3f}{out['ca4']:7.3f}"
                  f"{out['sc8']:7.3f}{out['ca8']:7.3f}{md:+9.3f}{p:6.3f}{star}")
        if rows:
            print("-" * 88)
            print(f"conf-aware@4 - SC@8: mean {np.mean(d48):+.4f}, "
                  f"CA@4 significantly beats SC@8 (half budget) in {sig48}/{rows} cells")
        return

    print(f"{'tag':26s}{'n':>5s}{'SC@8':>7s}{'confAware':>10s}{'Δ':>7s}{'CI95':>16s}{'p':>7s}")
    print("-" * 82)
    dd, sig, rows = [], 0, 0
    for tag in args.tags:
        scf = EXP_ROOT / "sampleconf" / f"{tag}_sc.json"
        lf = EXP_ROOT / "labels" / f"{tag}.json"
        if not (scf.exists() and lf.exists()):
            print(f"{tag}: missing"); continue
        recs = json.loads(scf.read_text()); labs = json.loads(lf.read_text())
        y, F, SCV = [], [], []
        for r in recs:
            if r["id"] not in labs or not r["samples"]:
                continue
            F.append(feats(r["samples"])); SCV.append(vote_only(r["samples"]))
            y.append(labs[r["id"]])
        y = np.array(y)
        if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0:
            print(f"{tag}: degenerate (n={len(y)}, pos={int(y.sum())})"); continue
        F = clean(np.array(F)); SCV = np.array(SCV)
        conf_aware = oof(F, y)
        md, lo, hi, p = boot(y, conf_aware, SCV)
        dd.append(md); sig += (p < 0.05); rows += 1
        star = "*" if p < 0.05 else " "
        print(f"{tag:26s}{len(y):5d}{au(y,SCV):7.3f}{au(y,conf_aware):10.3f}{md:+7.3f}"
              f"  [{lo:+.3f},{hi:+.3f}]{p:6.3f}{star}")
    if rows:
        print("-" * 82)
        print(f"conf-aware - SC@8: mean {np.mean(dd):+.4f}, sig > SC@8 in {sig}/{rows} cells")


if __name__ == "__main__":
    main()
