"""Learned-fusion budget frontier: does fusing confidence-dynamics with SC@k
BEAT vanilla SC at matched budget, and can fused@4 beat SC@8?

At each budget k in {2,4,8}: features = SC@k rich stats (from sampleconf answers)
+ per-sample confidence-dynamics aggregates over the k samples + primary-chain
CDYN (cached). Logistic head, 5-seed CV. Compare to vanilla SC@k (vote fraction)
and to SC@8. This is the robust (learned) version of confidence-aware voting.
"""
from __future__ import annotations

import argparse
import json
import math
import numpy as np
from collections import Counter

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


def samp_conf(s):
    lps = [x["lp"] for x in s.get("inter", []) if x["lp"] is not None]
    if not lps:
        return -10.0, -10.0, 0.0
    slope = np.polyfit(np.arange(len(lps)), lps, 1)[0] if len(lps) >= 2 else 0.0
    return float(np.mean(lps)), float(lps[-1]), float(slope)


def cluster(ans):
    reps = []; cl = []
    for i, a in ans:
        placed = False
        for ci, r in enumerate(reps):
            if _eq(a, r): cl[ci].append(i); placed = True; break
        if not placed: reps.append(a); cl.append([i])
    return cl


def feats_k(samps, k):
    ss = samps[:k]
    ans = [(i, s["final"]) for i, s in enumerate(ss) if s["final"] is not None]
    if not ans:
        return [0, 0, 0, -10, -10, 0, 1.0]
    means = np.array([samp_conf(ss[i])[0] for i, _ in ans])
    finals = np.array([samp_conf(ss[i])[1] for i, _ in ans])
    slopes = np.array([samp_conf(ss[i])[2] for i, _ in ans])
    w = np.exp(means - means.max()); w = w / w.sum()
    cl = cluster(ans)
    cnt = [len(c) for c in cl]
    idxmap = {i: j for j, (i, _) in enumerate(ans)}
    mass = [sum(w[idxmap[i]] for i in c) for c in cl]
    top = int(np.argmax(mass))
    vote = max(cnt) / len(ans)
    ent = -sum((c / len(ans)) * math.log(c / len(ans)) for c in cnt)
    conf_wtd = mass[top]
    mean_conf = float(np.mean(means)); mean_final = float(np.mean(finals))
    top_conf = float(np.mean([finals[idxmap[i]] for i in cl[int(np.argmax(cnt))]]))
    return [vote, conf_wtd, -ent, mean_conf, mean_final, top_conf, float(np.mean(slopes))]


def vote_only(samps, k):
    ans = [(i, s["final"]) for i, s in enumerate(samps[:k]) if s["final"] is not None]
    if not ans: return 0.0
    return max(len(c) for c in cluster(ans)) / len(ans)


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
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    au = lambda y, s: roc_auc_score(y, clean(np.asarray(s)).ravel())
    print(f"{'tag':22s}{'n':>4s}{'SC@4':>7s}{'Fus@4':>7s}{'SC@8':>7s}{'Fus@8':>7s}"
          f"{'F4-SC8':>8s}{'p':>6s}{'F8-SC8':>8s}{'p8':>6s}")
    print("-" * 84)
    d4, d8, s4, s8, rows = [], [], 0, 0, 0
    saved = {}
    for tag in args.tags:
        scf = EXP_ROOT / "sampleconf" / f"{tag}_sc.json"; lf = EXP_ROOT / "labels" / f"{tag}.json"
        cdf = EXP_ROOT / "cdyn" / f"{tag}.json"
        if not (scf.exists() and lf.exists()):
            continue
        recs = json.loads(scf.read_text()); labs = json.loads(lf.read_text())
        cd = json.loads(cdf.read_text()) if cdf.exists() else {}
        rr = [r for r in recs if r["id"] in labs and r["samples"]]
        y = np.array([labs[r["id"]] for r in rr])
        if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0:
            continue
        prim = np.array([cd[r["id"]]["cdyn"] if r["id"] in cd else [0]*6 for r in rr])
        res = {}
        for k in [4, 8]:
            F = clean(np.hstack([np.array([feats_k(r["samples"], k) for r in rr]), clean(prim)]))
            res[f"fus{k}"] = oof(F, y)
            res[f"sc{k}"] = np.array([vote_only(r["samples"], k) for r in rr])
        sc8 = res["sc8"]
        md4, lo4, hi4, p4 = boot(y, res["fus4"], sc8)
        md8, lo8, hi8, p8 = boot(y, res["fus8"], sc8)
        d4.append(md4); d8.append(md8); s4 += (p4 < 0.05 and md4 > 0); s8 += (p8 < 0.05 and md8 > 0); rows += 1
        st4 = "*" if (p4 < 0.05 and md4 > 0) else " "
        st8 = "*" if (p8 < 0.05 and md8 > 0) else " "
        print(f"{tag:22s}{len(y):4d}{au(y,res['sc4']):7.3f}{au(y,res['fus4']):7.3f}"
              f"{au(y,sc8):7.3f}{au(y,res['fus8']):7.3f}{md4:+8.3f}{p4:5.2f}{st4}{md8:+8.3f}{p8:5.2f}{st8}")
        saved[tag] = {"n": len(y), "sc4": au(y, res['sc4']), "fus4": au(y, res['fus4']),
                      "sc8": au(y, sc8), "fus8": au(y, res['fus8']), "d4": md4, "p4": p4, "d8": md8, "p8": p8}
    if rows:
        print("-" * 84)
        print(f"Fus@4 - SC@8: mean {np.mean(d4):+.4f}, Fus@4 sig-beats SC@8 (HALF budget) {s4}/{rows}")
        print(f"Fus@8 - SC@8: mean {np.mean(d8):+.4f}, Fus@8 sig-beats SC@8 (matched)     {s8}/{rows}")
    if args.out:
        json.dump(saved, open(args.out, "w"), indent=2); print("saved", args.out)


if __name__ == "__main__":
    main()
