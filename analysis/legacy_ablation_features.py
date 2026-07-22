"""ASD feature ablation (answers the 'patchwork' critique).

Feature index order (from cache):
 0 agree_frac  1 last_half_agree  2 conv_frac  3 flip_rate
 4 n_distinct  5 inter_entropy    6 none_frac  7 final_stable_run

Groups:
  agreement    = {0,1,7}
  oscillation  = {3,4,5}   (the novel centerpiece)
  convergence  = {2}
  failure      = {6}

We report AUROC (5-seed CV logistic) for:
  logprob-only  (2 cheap logprob feats)
  +each single ASD group (on top of logprob)
  full ASD (no logprob) ; full ASD + logprob (ours)
  leave-one-group-out from (full ASD + logprob)
  each ASD group ALONE (no logprob)
Plus mean logistic coefficients (sign/magnitude) on standardized features.
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
from acd import baselines as S

FEAT_NAMES = ["agree_frac", "last_half_agree", "conv_frac", "flip_rate",
              "n_distinct", "inter_entropy", "none_frac", "final_stable_run"]
GROUPS = {"agreement": [0, 1, 7], "oscillation": [3, 4, 5],
          "convergence": [2], "failure": [6]}


def clean(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        m = np.nanmin(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0
        x = np.where(np.isfinite(x), x, m)
    return x


def cv_auroc(X, y, seeds):
    if X.shape[1] == 0:
        return float("nan")
    accum = np.zeros(len(y))
    for s in seeds:
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            oof[te] = c.predict_proba(X[te])[:, 1]
        accum += oof
    return roc_auc_score(y, accum / len(seeds))


def load(tag):
    gen = json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{tag}.json").read_text())
    feats = json.loads((EXP_ROOT / "feats" / f"{tag}.json").read_text())
    y, F, LP = [], [], []
    for g in gen:
        if g["id"] not in labs or g["id"] not in feats:
            continue
        y.append(labs[g["id"]]); F.append(feats[g["id"]]["feat"])
        LP.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
    return np.array(y), clean(np.array(F)), clean(np.array(LP))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 7, 13, 42, 100])
    args = ap.parse_args()

    configs = {}
    # built lazily per dataset; store rows
    rows = {}
    coef_accum = {n: [] for n in FEAT_NAMES}

    def add(name, val, d):
        rows.setdefault(name, {})[d] = val

    for tag in args.tags:
        try:
            y, F, LP = load(tag)
        except FileNotFoundError:
            continue
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        d = tag.split("_")[0]
        add("logprob_only", cv_auroc(LP, y, args.seeds), d)
        for gname, idx in GROUPS.items():
            add(f"logprob+{gname}", cv_auroc(np.hstack([LP, F[:, idx]]), y, args.seeds), d)
        for gname, idx in GROUPS.items():
            add(f"{gname}_alone", cv_auroc(F[:, idx], y, args.seeds), d)
        add("ASD_full(no lp)", cv_auroc(F, y, args.seeds), d)
        add("ASD_full+logprob(OURS)", cv_auroc(np.hstack([F, LP]), y, args.seeds), d)
        for gname, idx in GROUPS.items():
            keep = [i for i in range(8) if i not in idx]
            add(f"OURS -{gname}", cv_auroc(np.hstack([F[:, keep], LP]), y, args.seeds), d)
        # coefficients (standardized) on full ASD, single seed
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(F, y)
        w = clf.named_steps["logisticregression"].coef_[0]
        for n, wi in zip(FEAT_NAMES, w):
            coef_accum[n].append(wi)

    ds = list(dict.fromkeys(d for r in rows.values() for d in r))
    order = (["logprob_only"] + [f"logprob+{g}" for g in GROUPS]
             + [f"{g}_alone" for g in GROUPS]
             + ["ASD_full(no lp)", "ASD_full+logprob(OURS)"]
             + [f"OURS -{g}" for g in GROUPS])
    print(f"\n=== ASD ablation: AUROC (5-seed CV) ===")
    hdr = f"{'config':26s}" + "".join(f"{d[:8]:>9s}" for d in ds) + f"{'AVG':>8s}"
    print(hdr); print("-" * len(hdr))
    for name in order:
        if name not in rows:
            continue
        vals = [rows[name].get(d, float('nan')) for d in ds]
        avg = np.nanmean(vals)
        print(f"{name:26s}" + "".join(f"{v:9.3f}" for v in vals) + f"{avg:8.3f}")

    print("\n=== mean standardized logistic coefficients (ASD-only) ===")
    print("(positive => higher value predicts CORRECT)")
    for n in FEAT_NAMES:
        c = np.mean(coef_accum[n]) if coef_accum[n] else float('nan')
        grp = next(g for g, idx in GROUPS.items() if FEAT_NAMES.index(n) in idx)
        print(f"  {n:18s} [{grp:11s}] {c:+.3f}")


if __name__ == "__main__":
    main()
