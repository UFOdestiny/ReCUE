"""Formal cost-aware budget allocation (review3 priority 4).

Problem: given N questions and a total sample budget B (avg samples/question),
decide how many self-consistency samples each question gets, to maximize final
majority-vote accuracy. A per-question 'uncertainty' score decides who gets more.

Policies (all use 1 sample as base, then spend extra where a scorer says uncertain):
  uniform@k          : everyone gets k (the standard SC baseline)
  random             : random allocation at matched budget
  entropy_guided     : score = single-chain forced-answer confidence (our cheap signal)
  ptrue_guided       : score = 1 - p_true
  answerconv_guided  : score = 1 - answer-convergence-confidence
  dynamics_guided    : score = 1 - ours(conv+cdyn) OOF prob    (OUR method)
  oracle             : spend on the ones that flip to correct with more samples

Reports final accuracy vs average budget (samples/question). Shows dynamics-guided
reaches SC@8 accuracy at much lower average budget.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from acd.env import EXP_ROOT
from acd import data as dv
from acd import baselines as S

SEEDS = [2026, 7, 13]


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


def maj_correct(ans_list, k, gold, gold_raw):
    a = [x for x in ans_list[:k] if x is not None]
    if not a:
        return 0
    top = Counter(a).most_common(1)[0][0]
    return dv.verify({"gold": gold, "gold_raw": gold_raw}, "\\boxed{%s}" % top)


def alloc_accuracy(recs, order, budget_avg, kmax=8):
    """order: question indices sorted MOST-uncertain first. budget_avg: avg samples/q.
    Everyone gets 1; the extra (budget_avg-1)*N sample-units go to the most-uncertain,
    each bumped to kmax (spend greedily)."""
    N = len(recs)
    extra_units = int(round((budget_avg - 1) * N))
    per_q = np.ones(N, dtype=int)
    i = 0
    while extra_units > 0 and i < N:
        qi = order[i]
        add = min(kmax - per_q[qi], extra_units)
        per_q[qi] += add; extra_units -= add; i += 1
    correct = 0
    for qi, r in enumerate(recs):
        correct += maj_correct(r["ans"], per_q[qi], r["gold"], r["gold_raw"])
    return correct / N, per_q.mean()


def load(tag):
    gen = {g["id"]: g for g in json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())}
    labs = json.loads((EXP_ROOT / "labels" / f"{tag}.json").read_text())
    sampans = json.loads((EXP_ROOT / "sampans" / f"{tag}.json").read_text())
    cdyn = json.loads((EXP_ROOT / "cdyn" / f"{tag}.json").read_text())
    ptf = EXP_ROOT / "ptrue" / f"{tag}_ptrue.json"
    ptrue = json.loads(ptf.read_text()) if ptf.exists() else None
    recs, y, CONV, CDYN, SEQ, conf1, pt = [], [], [], [], [], [], []
    for i, g in gen.items():
        if i not in labs or i not in cdyn or i not in sampans:
            continue
        recs.append({"ans": sampans[i], "gold": g["gold"], "gold_raw": g.get("gold_raw")})
        y.append(labs[i]); CONV.append(cdyn[i]["conv"]); CDYN.append(cdyn[i]["cdyn"])
        SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        conf1.append(cdyn[i]["cdyn"][1])  # last_lp = cheap single-chain confidence
        pt.append(ptrue.get(i, 0.5) if ptrue else np.nan)
    return recs, np.array(y), clean(np.array(CONV)), clean(np.array(CDYN)), clean(np.array(SEQ)), np.array(conf1), np.array(pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    budgets = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    for tag in args.tags:
        try:
            recs, y, CONV, CDYN, SEQ, conf1, pt = load(tag)
        except FileNotFoundError:
            continue
        if len(y) < 30:
            continue
        N = len(recs)
        # scorers: higher => more UNCERTAIN => gets more samples
        dyn = 1 - oof(np.hstack([CONV, CDYN, SEQ]), y)
        aconv = 1 - oof(CONV, y)
        unc = {
            "dynamics": np.argsort(-dyn),
            "answerconv": np.argsort(-aconv),
            "confidence1": np.argsort(conf1),     # low last_lp = uncertain
        }
        if pt is not None and np.isfinite(pt).any():
            unc["ptrue"] = np.argsort(pt)          # low p_true = uncertain
        rng = np.random.RandomState(0)
        print(f"\n=== {tag} (N={N}, acc@1={np.mean([maj_correct(r['ans'],1,r['gold'],r['gold_raw']) for r in recs]):.3f}, "
              f"acc@8={np.mean([maj_correct(r['ans'],8,r['gold'],r['gold_raw']) for r in recs]):.3f}) ===")
        header = "budget " + "".join(f"{k:>13s}" for k in list(unc.keys()) + ["uniform", "random"])
        print(header)
        for b in budgets:
            row = f"{b:5.1f}x "
            for k, order in unc.items():
                acc, avgk = alloc_accuracy(recs, list(order), b)
                row += f"{acc:13.3f}"
            # uniform: everyone gets round(b)
            ku = int(round(b))
            accu = np.mean([maj_correct(r["ans"], ku, r["gold"], r["gold_raw"]) for r in recs])
            # random
            accr, _ = alloc_accuracy(recs, list(rng.permutation(N)), b)
            row += f"{accu:13.3f}{accr:13.3f}"
            print(row)


if __name__ == "__main__":
    main()
