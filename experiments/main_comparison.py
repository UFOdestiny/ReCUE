"""Master comparison table with confidence-dynamics integrated (all 6 families).

Uses conf caches (per-cut neutral answer + forced-answer logprob) for our method
and the answer-convergence named baseline; gen caches for logprob/self-consistency;
ptrue caches where available.

Methods:
  BASELINES (1x):    mean_logprob, self_certainty, deepconf_bottom
  BASELINE (1fwd):   p_true
  BASELINE (1x):     answer_convergence  (CONV feature-set logreg — prior-art proxy)
  OURS (1x):         conv+confidence_dynamics (CONV + CDYN + seq-logprob), 5-seed CV
  MULTI (kx):        self_consistency@{2,4,8}
Report AUROC + AURC, per dataset, macro, and per-family Δ(ours - answer_convergence).
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
from acd import baselines as S
from acd import metrics as UQ
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


def conv_cdyn(rec):
    inter = rec["intermediate"]; neu = [x["neutral"] for x in inter]
    lp = [x.get("neutral_lp") for x in inter]
    n = len(neu); final = neu[-1] if neu else None; half = n // 2
    agree = [1.0 if (a is not None and final is not None and _eq(a, final)) else 0.0 for a in neu]
    af = np.mean(agree) if n else 0.0
    lh = np.mean(agree[half:]) if n - half > 0 else af
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
    lpv = [v for v in lp if v is not None]
    slope = np.polyfit(np.arange(len(lpv)), lpv, 1)[0] if len(lpv) >= 2 else 0.0
    fc = -10.0
    for a, v in zip(neu, lp):
        if a is not None and final is not None and _eq(a, final) and v is not None:
            fc = v; break
    CDYN = [np.mean(lpv) if lpv else -10, lpv[-1] if lpv else -10, np.min(lpv) if lpv else -10,
            slope, fc, np.std(lpv) if len(lpv) > 1 else 0.0]
    return CONV, CDYN


def sc(ans, k):
    a = [x for x in ans[:k] if x is not None]
    return Counter(a).most_common(1)[0][1] / len(a) if a else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    R = {}
    for tag in args.tags:
        cf = EXP_ROOT / "conf" / f"{tag}_conf.json"
        lf = EXP_ROOT / "labels" / f"{tag}.json"
        gf = EXP_ROOT / "gen" / f"{tag}.json"
        saf = EXP_ROOT / "sampans" / f"{tag}.json"
        if not (cf.exists() and lf.exists() and gf.exists()):
            continue
        cdf = EXP_ROOT / "cdyn" / f"{tag}.json"
        if not cdf.exists():
            continue
        cdyn_cache = json.loads(cdf.read_text())
        recs = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
        gen = {g["id"]: g for g in json.loads(gf.read_text())}
        sampans = json.loads(saf.read_text()) if saf.exists() else {}
        ptf = EXP_ROOT / "ptrue" / f"{tag}_ptrue.json"
        ptrue = json.loads(ptf.read_text()) if ptf.exists() else None
        y, CONV, CDYN, SEQ, sp_scores, pt, ans = [], [], [], [], defaultdict(list), [], []
        for r in recs:
            if not r["intermediate"] or r["id"] not in labs or r["id"] not in cdyn_cache:
                continue
            cv_, cd_ = cdyn_cache[r["id"]]["conv"], cdyn_cache[r["id"]]["cdyn"]
            CONV.append(cv_); CDYN.append(cd_)
            g = gen.get(r["id"], {})
            SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
            sp_scores["mean_logprob"].append(S.sig_mean_logprob(g))
            sp_scores["self_certainty"].append(S.sig_self_certainty(g))
            sp_scores["deepconf_bottom"].append(S.sig_deepconf_bottom(g))
            pt.append(ptrue.get(r["id"], 0.5) if ptrue else np.nan)
            ans.append(sampans.get(r["id"], []))
            y.append(labs[r["id"]])
        y = np.array(y)
        if len(y) < 20 or y.sum() == 0 or (1 - y).sum() == 0:
            continue
        CONV = np.array(CONV); CDYN = np.array(CDYN); SEQ = clean(np.array(SEQ))
        entry = {"n": len(y), "acc": float(y.mean()), "au": {}, "ar": {}}

        def rec_(name, s):
            s = clean(np.asarray(s)).ravel()
            entry["au"][name] = roc_auc_score(y, s); entry["ar"][name] = UQ.aurc(s, y)

        for m in sp_scores:
            rec_(m, np.array(sp_scores[m]))
        if ptrue is not None:
            rec_("p_true", np.array(pt))
        rec_("answer_convergence", oof(CONV, y))
        rec_("ours(conv+confdyn)", oof(np.hstack([CONV, CDYN, SEQ]), y))
        for k in [2, 4, 8]:
            if ans and any(ans):
                rec_(f"self_consistency@{k}", np.array([sc(a, k) for a in ans]))
        R[tag] = entry

    order = ["mean_logprob", "self_certainty", "deepconf_bottom", "p_true",
             "answer_convergence", "ours(conv+confdyn)",
             "self_consistency@2", "self_consistency@4", "self_consistency@8"]
    ds = list(R.keys())
    print(f"\n=== AUROC (n={len(ds)} cells) ===")
    print(f"{'method':22s}{'macro':>8s}{'#best':>7s}")
    print("-" * 40)
    best_count = defaultdict(int)
    single_pass = [m for m in order if not m.startswith("self_cons")]
    for d in ds:
        vals = {m: R[d]["au"].get(m, -1) for m in single_pass}
        bm = max(vals, key=vals.get); best_count[bm] += 1
    for m in order:
        vals = [R[d]["au"][m] for d in ds if m in R[d]["au"]]
        if not vals:
            continue
        print(f"{m:22s}{np.mean(vals):8.3f}{best_count.get(m,0):7d}")
    # ours vs answer_convergence delta, per family
    print("\nΔ ours - answer_convergence, per family (AUROC):")
    fam = defaultdict(list)
    for d in ds:
        if "ours(conv+confdyn)" in R[d]["au"] and "answer_convergence" in R[d]["au"]:
            f = "_".join(d.split("_")[1:-1])
            fam[f].append(R[d]["au"]["ours(conv+confdyn)"] - R[d]["au"]["answer_convergence"])
    for f, v in sorted(fam.items()):
        print(f"  {f:16s} {np.mean(v):+.3f} (n={len(v)})")
    alld = [x for v in fam.values() for x in v]
    print(f"  {'OVERALL':16s} {np.mean(alld):+.3f}")
    if args.out:
        json.dump({d: R[d] for d in ds}, open(args.out, "w"), indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
