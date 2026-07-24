"""Confound control: is answer-stability an INDEPENDENT correctness signal, or
just a proxy for difficulty / trace length / final-answer logprob?

Reviewer's decisive test. We report, per dataset-model:
  (A) incremental AUROC: [logprob+entropy+length(+difficulty)] vs +stability
  (B) partial signal: AUROC of stability WITHIN length terciles (pooled)
  (C) correlation of stability with length & logprob (should be < 1 to be independent)
Uses only cached features/labels (offline).
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

from recue.env import EXP_ROOT
from recue import baselines as S

FEAT = ["agree_frac", "last_half_agree", "conv_frac", "flip_rate",
        "n_distinct", "inter_entropy", "none_frac", "final_stable_run"]
SEEDS = [2026, 7, 13, 42, 100]


def clean(x):
    x = np.asarray(x, float)
    if not np.isfinite(x).all():
        m = np.nanmin(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0
        x = np.where(np.isfinite(x), x, m)
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
    return roc_auc_score(y, acc / len(SEEDS)), acc / len(SEEDS)


def load(tag):
    gen = json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{tag}.json").read_text())
    feats = json.loads((EXP_ROOT / "feats" / f"{tag}.json").read_text())
    y, F, lp, length, dyn = [], [], [], [], []
    for g in gen:
        i = g["id"]
        if i not in labs or i not in feats:
            continue
        y.append(labs[i]); F.append(feats[i]["feat"]); dyn.append(feats[i]["dyn_scalar"])
        lp.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        length.append(g.get("n_gen_tokens", 0))
    return (np.array(y), clean(np.array(F)), clean(np.array(lp)),
            np.array(length, float), clean(np.array(dyn)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    print(f"{'setting':22s}{'lp+len':>8s}{'+stab':>8s}{'Δ':>7s}"
          f"{'stab|len(within)':>17s}{'r(stab,len)':>12s}{'r(stab,lp)':>11s}")
    print("-" * 85)
    agg = defaultdict(list)
    for tag in args.tags:
        try:
            y, F, lp, length, dyn = load(tag)
        except FileNotFoundError:
            continue
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        loglen = np.log1p(length).reshape(-1, 1)
        base = np.hstack([lp, loglen])            # logprob+entropy+length
        a_base, _ = cv(base, y)
        a_aug, _ = cv(np.hstack([base, F]), y)     # + full stability features
        # within-length-tercile AUROC of the stability scalar (pooled, difficulty-controlled)
        terc = np.quantile(length, [1/3, 2/3])
        buckets = np.digitize(length, terc)
        within_scores = np.full(len(y), np.nan)
        for b in np.unique(buckets):
            m = buckets == b
            if y[m].sum() > 0 and (1 - y[m]).sum() > 0:
                # rank stability within bucket -> pooled AUROC via bucket-standardized score
                within_scores[m] = (dyn[m] - dyn[m].mean()) / (dyn[m].std() + 1e-9)
        ok = ~np.isnan(within_scores)
        a_within = roc_auc_score(y[ok], within_scores[ok]) if (y[ok].sum() and (1-y[ok]).sum()) else float('nan')
        r_len = np.corrcoef(dyn, length)[0, 1]
        r_lp = np.corrcoef(dyn, lp[:, 0])[0, 1]
        d = "_".join(tag.split("_")[:2])
        print(f"{d:22s}{a_base:8.3f}{a_aug:8.3f}{a_aug-a_base:+7.3f}"
              f"{a_within:17.3f}{r_len:12.3f}{r_lp:11.3f}")
        agg["base"].append(a_base); agg["aug"].append(a_aug); agg["within"].append(a_within)
        agg["rlen"].append(r_len); agg["rlp"].append(r_lp)
    print("-" * 85)
    print(f"{'MEAN':22s}{np.mean(agg['base']):8.3f}{np.mean(agg['aug']):8.3f}"
          f"{np.mean(agg['aug'])-np.mean(agg['base']):+7.3f}"
          f"{np.nanmean(agg['within']):17.3f}{np.nanmean(agg['rlen']):12.3f}{np.nanmean(agg['rlp']):11.3f}")
    print("\nInterpretation: Δ>0 => stability adds signal beyond logprob+entropy+length.")
    print("within-length AUROC>>0.5 => not merely a length/difficulty proxy.")
    print("|r|<~0.5 => stability is not collinear with length/logprob.")


if __name__ == "__main__":
    main()
