"""Within-problem AUROC (review2's decisive control).

Controls difficulty/gold/domain/wording by asking, WITHIN each problem that has
both correct and wrong sampled traces: does a per-trace score rank the correct
trace above the wrong one? Macro-average per-problem AUROC.

Per-trace scores available WITHOUT re-probing (from k=8 sample cache):
  - sample_meanlp  : mean token logprob of that trace       (logprob baseline)
  - agree_with_mode: 1 if this trace's answer == the k-sample modal answer, else 0
                     (a cheap cross-sample 'stability' proxy)
We compare these head-to-head under the within-problem protocol. (A full
per-sample intra-trace stability probe is a separate GPU job; this establishes
the protocol and the logprob/agreement references first.)
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from collections import Counter

from acd.env import EXP_ROOT
from sklearn.metrics import roc_auc_score


def per_problem_auroc(scores_correct, scores_wrong):
    """AUROC that a random correct trace outscores a random wrong trace (=Mann-Whitney)."""
    c = np.asarray(scores_correct, float); w = np.asarray(scores_wrong, float)
    n = len(c) * len(w)
    if n == 0:
        return None
    wins = 0.0
    for a in c:
        wins += np.sum(a > w) + 0.5 * np.sum(a == w)
    return wins / n


def load(tag):
    gen = json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{tag}.json").read_text())
    sampans = json.loads((EXP_ROOT / "sampans" / f"{tag}.json").read_text())
    return gen, labs, sampans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    print(f"{'setting':22s}{'#probs':>7s}{'wp_logprob':>11s}{'wp_agree':>10s}")
    print("-" * 50)
    agg = {"lp": [], "ag": []}
    for tag in args.tags:
        try:
            gen, labs, sampans = load(tag)
        except FileNotFoundError:
            continue
        # need per-sample correctness: recompute from sample text via math_verify
        from acd import data as dv
        lp_aurocs, ag_aurocs = [], []
        n_used = 0
        for g in sampans and gen or []:
            i = g["id"]
            ans = sampans.get(i, [])
            meanlp = g.get("sample_meanlp") or []
            if len(ans) < 2:
                continue
            # per-sample correctness
            corr = []
            for a in ans:
                if a is None:
                    corr.append(0); continue
                corr.append(dv.verify({"gold": g["gold"], "gold_raw": g.get("gold_raw")},
                                      "\\boxed{%s}" % a))
            corr = np.array(corr)
            if corr.sum() == 0 or (1 - corr).sum() == 0:
                continue  # need both correct & wrong within this problem
            n_used += 1
            # modal answer agreement (stability proxy)
            valid = [a for a in ans if a is not None]
            mode = Counter(valid).most_common(1)[0][0] if valid else None
            agree = np.array([1.0 if a == mode else 0.0 for a in ans])
            # logprob per sample (impute missing with min)
            ml = np.array([m if m is not None else -20.0 for m in meanlp[:len(ans)]], float)
            if len(ml) < len(ans):
                ml = np.concatenate([ml, np.full(len(ans) - len(ml), -20.0)])
            ci = np.where(corr == 1)[0]; wi = np.where(corr == 0)[0]
            a_lp = per_problem_auroc(ml[ci], ml[wi])
            a_ag = per_problem_auroc(agree[ci], agree[wi])
            if a_lp is not None:
                lp_aurocs.append(a_lp)
            if a_ag is not None:
                ag_aurocs.append(a_ag)
        if not lp_aurocs:
            continue
        d = "_".join(tag.split("_")[:2])
        mlp, mag = np.mean(lp_aurocs), np.mean(ag_aurocs)
        print(f"{d:22s}{n_used:7d}{mlp:11.3f}{mag:10.3f}")
        agg["lp"].append(mlp); agg["ag"].append(mag)
    print("-" * 50)
    print(f"{'MEAN':22s}{'':>7s}{np.mean(agg['lp']):11.3f}{np.mean(agg['ag']):10.3f}")
    print("\nwp = within-problem macro AUROC (only problems with both correct & wrong traces).")
    print("Controls difficulty/gold/domain. >0.5 => signal separates correct vs wrong on SAME problem.")


if __name__ == "__main__":
    main()
