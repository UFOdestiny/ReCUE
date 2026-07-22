"""SOTA judge-free UQ comparison using cached correctness labels (fast).

Baselines (single-pass, judge-free):
  mean_logprob, mean_entropy, self_certainty (Kang'25), DeepConf-tail/bottom (Fu'25)
Optional single extra forward: P(True) (Kadavath'22)  [loaded from ptrue/ cache if present]
Multi-sample upper anchor: self_consistency@k, semantic answer_entropy@k
Ours: dyn_scalar, dyn+logprob (5-seed CV)
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from acd.env import EXP_ROOT
from acd import baselines as S
from acd import features as D
from acd import metrics as M

FEAT = ["agree_frac", "last_half_agree", "conv_frac", "flip_rate",
        "n_distinct", "inter_entropy", "none_frac", "final_stable_run"]


def clean(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        m = np.nanmin(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0
        x = np.where(np.isfinite(x), x, m)
    return x


def cv(X, y, seeds):
    acc = np.zeros(len(y))
    for s in seeds:
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            oof[te] = c.predict_proba(X[te])[:, 1]
        acc += oof
    return acc / len(seeds)


def _sc_from_answers(ans, k):
    from collections import Counter
    a = [x for x in ans[:k] if x is not None]
    if not a:
        return 0.0
    c = Counter(a); top, n = c.most_common(1)[0]
    return n / len(a)


def _se_from_answers(ans, k):
    import math
    from collections import Counter
    a = [x for x in ans[:k] if x is not None]
    if not a:
        return -1e9
    c = Counter(a); tot = sum(c.values())
    return -(-sum((v / tot) * math.log(v / tot) for v in c.values()))


def load(tag, k_sc, seeds):
    gen = json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{tag}.json").read_text())
    sampans = json.loads((EXP_ROOT / "sampans" / f"{tag}.json").read_text())
    feats = json.loads((EXP_ROOT / "feats" / f"{tag}.json").read_text())
    ptrue_path = EXP_ROOT / "ptrue" / f"{tag}_ptrue.json"
    ptrue = json.loads(ptrue_path.read_text()) if ptrue_path.exists() else None

    y, F, sc, se = [], [], [], []
    scores = {m: [] for m in ["mean_logprob", "mean_entropy", "self_certainty",
                              "deepconf_tail", "deepconf_bottom"]}
    dyn_s, pt = [], []
    for g in gen:
        if g["id"] not in labs or g["id"] not in feats:
            continue
        y.append(labs[g["id"]])
        for m in scores:
            scores[m].append(S.SINGLE_PASS[m](g))
        F.append(feats[g["id"]]["feat"])
        dyn_s.append(feats[g["id"]]["dyn_scalar"])
        ans = sampans.get(g["id"], [])
        sc.append(_sc_from_answers(ans, k_sc)); se.append(_se_from_answers(ans, k_sc))
        pt.append(ptrue.get(g["id"], 0.5) if ptrue else np.nan)
    y = np.array(y)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return None
    F = clean(np.array(F))
    row = {"n": len(y), "acc": float(y.mean()), "auroc": {}, "aurc": {}}

    def rec(name, s):
        s = clean(s)
        row["auroc"][name] = roc_auc_score(y, s)
        row["aurc"][name] = M.aurc(s, y)

    for m in scores:
        rec(m, np.array(scores[m]))
    if ptrue is not None:
        rec("p_true(1fwd)", np.array(pt))
    rec("dyn_scalar(ours)", np.array(dyn_s))
    rec("dyn+logprob(ours)", cv(np.hstack([F, np.array(scores["mean_logprob"]).reshape(-1, 1),
                                            np.array(scores["mean_entropy"]).reshape(-1, 1)]), y, seeds))
    rec(f"self_consistency@{k_sc}", np.array(sc))
    rec(f"answer_entropy@{k_sc}", np.array(se))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--k-sc", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 7, 13, 42, 100])
    args = ap.parse_args()

    R = {}
    for t in args.tags:
        try:
            r = load(t, args.k_sc, args.seeds)
        except FileNotFoundError:
            r = None
        if r:
            R[t] = r
    ds = list(R.keys())
    order = ["mean_logprob", "mean_entropy", "self_certainty", "deepconf_tail",
             "deepconf_bottom", "p_true(1fwd)", "dyn_scalar(ours)", "dyn+logprob(ours)",
             f"self_consistency@{args.k_sc}", f"answer_entropy@{args.k_sc}"]
    for metric in ["auroc", "aurc"]:
        print(f"\n=== {metric.upper()} ({'higher' if metric=='auroc' else 'lower'} better) ===")
        hdr = f"{'method':20s}" + "".join(f"{d.split('_')[0][:8]:>9s}" for d in ds) + f"{'AVG':>8s}"
        print(hdr); print("-" * len(hdr))
        for m in order:
            vals = [R[d][metric].get(m) for d in ds]
            if any(v is None for v in vals):
                continue
            tail = "  <-multi" if ("@" in m) else ("  <-1fwd" if "1fwd" in m else "")
            print(f"{m:20s}" + "".join(f"{v:9.3f}" for v in vals) + f"{np.mean(vals):8.3f}" + tail)
    print("\nacc:", {d.split('_')[0]: round(R[d]["acc"], 3) for d in ds})


if __name__ == "__main__":
    main()
