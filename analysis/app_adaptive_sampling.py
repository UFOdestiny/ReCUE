"""Application 2: confidence-guided adaptive self-consistency.

Single-chain dynamics confidence tells us WHERE to spend samples. Confident
questions keep 1 sample; uncertain ones get the full k samples (majority vote).
Offline simulation from k=8 caches.

Baselines:
  - SC@1  : greedy single answer                          (1 sample)
  - SC@k  : full self-consistency majority vote           (k samples)
  - random-route: spend k on a random rho-fraction        (matched budget)
  - ours  : spend k on the rho-fraction with LOWEST single-chain confidence
Report: final accuracy vs average #samples/question.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
import numpy as np

from acd.env import EXP_ROOT
from acd import data as dv
from acd import features as D


def maj_answer(answers):
    a = [x for x in answers if x is not None]
    if not a:
        return None
    return Counter(a).most_common(1)[0][0]


def correct(ans, r):
    if ans is None:
        return 0
    return dv.verify({"gold": r["gold"], "gold_raw": r.get("gold_raw")}, "\\boxed{%s}" % ans)


def load(tag):
    gen = json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())
    probe = {p["id"]: p for p in json.loads((EXP_ROOT / "probe" / f"{tag}_probe.json").read_text())}
    recs = []
    for g in gen:
        pr = probe.get(g["id"])
        if pr is None:
            continue
        from acd.baselines import sampled_answers
        recs.append({
            "samples_ans": sampled_answers(g),   # extracted answer per sample
            "conf": D.scalar_confidence(pr),      # single-chain dynamics confidence
            "gold": g["gold"], "gold_raw": g.get("gold_raw"),
        })
    return recs


def acc_sc(recs, k):
    return np.mean([correct(maj_answer(r["samples_ans"][:k]), r) for r in recs])


def acc_routed(recs, k, rho, order):
    """Route: the rho-fraction selected by `order` (list of indices) gets k samples,
    the rest get 1. Return (accuracy, avg_samples)."""
    n = len(recs)
    n_hard = int(round(rho * n))
    hard = set(order[:n_hard])
    accs, samples = [], []
    for i, r in enumerate(recs):
        kk = k if i in hard else 1
        accs.append(correct(maj_answer(r["samples_ans"][:kk]), r))
        samples.append(kk)
    return np.mean(accs), np.mean(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    for tag in args.tags:
        try:
            recs = load(tag)
        except FileNotFoundError:
            continue
        if not recs:
            continue
        n = len(recs)
        a1 = acc_sc(recs, 1)
        ak = acc_sc(recs, args.k)
        # order by ascending confidence => most uncertain first (ours)
        conf = np.array([r["conf"] for r in recs])
        ours_order = list(np.argsort(conf))
        print(f"\n=== {tag} (n={n}) ===")
        print(f"SC@1 acc={a1:.3f} (1.0 samp)   SC@{args.k} acc={ak:.3f} ({args.k}.0 samp)")
        print(f"{'rho':>5s} {'ours_acc':>9s} {'ours_samp':>10s} {'rand_acc(mean)':>14s}  {'ours-rand':>10s}")
        for rho in [0.2, 0.4, 0.6]:
            oa, osamp = acc_routed(recs, args.k, rho, ours_order)
            # random baseline averaged over 20 seeds
            ras = []
            for s in range(20):
                rng = np.random.RandomState(1000 + s)
                ras.append(acc_routed(recs, args.k, rho, list(rng.permutation(n)))[0])
            ra = float(np.mean(ras)); ra_sd = float(np.std(ras))
            frac_gain = (oa - a1) / (ak - a1) if ak > a1 else float('nan')
            print(f"{rho:5.1f} {oa:9.3f} {osamp:10.2f} {ra:8.3f}±{ra_sd:.3f}  {oa-ra:+9.3f}"
                  f"  (recovers {frac_gain*100:4.0f}% of SC gain)")


if __name__ == "__main__":
    main()
