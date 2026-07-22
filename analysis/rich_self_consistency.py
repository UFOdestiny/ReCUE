"""Beat SOTA by using the SAME k samples more fully than vanilla self-consistency.

Vanilla SC@8 = a single number (modal-answer vote fraction). But the 8 sampled
answers carry much more: the full answer distribution shape. We build a "rich SC"
feature set from the SAME 8 answers (no extra generation, matched 8x budget):
  vote_frac      : top-1 fraction (= vanilla SC@8)
  top2_margin    : (count1 - count2)/k
  neg_entropy    : -Shannon entropy of the answer distribution
  n_distinct     : distinct answers / k
  singleton_frac : fraction of answers appearing exactly once
  none_frac      : fraction of samples with no parseable answer
Then compare (5-seed CV AUROC, bootstrap CI vs vanilla SC@8):
  vanilla SC@8           (the SOTA)
  rich_SC (logreg)       (same 8 samples, richer features)
  rich_SC + ours(cdyn)   (adds single-chain answer-confidence dynamics)
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
from acd import baselines as S

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


def rich_feats(ans, k=8):
    a = ans[:k]
    valid = [x for x in a if x is not None]
    none_frac = 1.0 - len(valid) / max(len(a), 1)
    if not valid:
        return [0, 0, 0, 1.0, 0, none_frac]
    c = Counter(valid); counts = sorted(c.values(), reverse=True)
    tot = sum(counts)
    vote = counts[0] / len(valid)
    top2 = (counts[0] - (counts[1] if len(counts) > 1 else 0)) / len(valid)
    ent = -sum((v / tot) * math.log(v / tot) for v in counts)
    ndist = len(counts) / len(valid)
    singleton = sum(1 for v in counts if v == 1) / len(valid)
    return [vote, top2, -ent, ndist, singleton, none_frac]


def vote_only(ans, k=8):
    a = [x for x in ans[:k] if x is not None]
    return Counter(a).most_common(1)[0][1] / len(a) if a else 0.0


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
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    print(f"{'tag':24s}{'n':>5s}{'SC@8':>7s}{'richSC':>7s}{'rich+dyn':>9s}"
          f"{'Δrich':>7s}{'CI95(rich-SC)':>17s}{'p':>6s}")
    print("-" * 92)
    dr, drd, sig_r, sig_rd, rows = [], [], 0, 0, 0
    for tag in args.tags:
        cdf = EXP_ROOT / "cdyn" / f"{tag}.json"; lf = EXP_ROOT / "labels" / f"{tag}.json"
        gf = EXP_ROOT / "gen" / f"{tag}.json"; saf = EXP_ROOT / "sampans" / f"{tag}.json"
        if not all(p.exists() for p in [cdf, lf, gf, saf]): continue
        cd = json.loads(cdf.read_text()); labs = json.loads(lf.read_text())
        gen = {g["id"]: g for g in json.loads(gf.read_text())}; sa = json.loads(saf.read_text())
        y, RICH, SCV, CONV, CDYN, SEQ = [], [], [], [], [], []
        for i in sa:
            if i not in labs or i not in cd: continue
            RICH.append(rich_feats(sa[i], args.k)); SCV.append(vote_only(sa[i], args.k))
            CONV.append(cd[i]["conv"]); CDYN.append(cd[i]["cdyn"])
            g = gen.get(i, {}); SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
            y.append(labs[i])
        y = np.array(y)
        if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0: continue
        RICH = clean(np.array(RICH)); SCV = np.array(SCV)
        base = np.hstack([clean(np.array(CONV)), clean(np.array(CDYN)), clean(np.array(SEQ))])
        s_rich = oof(RICH, y)
        s_richdyn = oof(np.hstack([RICH, base]), y)
        au = lambda s: roc_auc_score(y, clean(np.asarray(s)).ravel())
        mdr, lor, hir, pr = boot(y, s_rich, SCV)
        mdrd, _, _, prd = boot(y, s_richdyn, SCV)
        dr.append(mdr); drd.append(mdrd); sig_r += (pr < 0.05); sig_rd += (prd < 0.05); rows += 1
        star = "*" if pr < 0.05 else " "
        print(f"{tag:24s}{len(y):5d}{au(SCV):7.3f}{au(s_rich):7.3f}{au(s_richdyn):9.3f}"
              f"{mdr:+7.3f}  [{lor:+.3f},{hir:+.3f}]{pr:6.3f}{star}")
    print("-" * 92)
    print(f"richSC - SC@8:      mean {np.mean(dr):+.4f}, sig > SC@8 in {sig_r}/{rows} cells")
    print(f"rich+dyn - SC@8:    mean {np.mean(drd):+.4f}, sig > SC@8 in {sig_rd}/{rows} cells")


if __name__ == "__main__":
    main()
