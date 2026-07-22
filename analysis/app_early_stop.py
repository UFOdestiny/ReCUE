"""Application 1: adaptive early-stopping from the stabilization signal.

Offline simulation using cached probe trajectories. Decision rule: stop at the
earliest probe cut whose answer has been STABLE for the last `patience`
consecutive probes; emit that answer. Token cost is approximated by the fraction
of the reasoning consumed at the stopping cut (cut/n_segs) x total gen tokens,
plus the cheap probe decodes up to that point.

We compare:
  - full   : run the whole chain (100% tokens), emit final answer  (acc upper ref)
  - fixed@f: truncate every chain at fraction f, emit that cut's answer
  - ours   : adaptive stop when stable for `patience` probes
Reported: accuracy retained vs average token fraction used.
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from acd.env import EXP_ROOT
from acd import data as dv
from acd.features import _eq


def emit_correct(ans, gold_raw, gold):
    if ans is None:
        return 0
    row = {"gold": gold, "gold_raw": gold_raw}
    return dv.verify(row, "\\boxed{%s}" % ans)


def load(tag):
    gen = {g["id"]: g for g in json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())}
    probe = json.loads((EXP_ROOT / "probe" / f"{tag}_probe.json").read_text())
    recs = []
    for pr in probe:
        g = gen.get(pr["id"])
        if g is None or not pr.get("intermediate"):
            continue
        recs.append({
            "inter": pr["intermediate"], "n_segs": pr["n_segs"],
            "final": pr["final_answer"], "gold": pr["gold"], "gold_raw": pr.get("gold_raw"),
            "ntok": g.get("n_gen_tokens", 0), "probe_ntok": pr.get("probe_ntok", []),
        })
    return recs


def eval_full(recs):
    acc = np.mean([emit_correct(r["final"], r["gold_raw"], r["gold"]) for r in recs])
    return acc, 1.0


def eval_fixed(recs, frac):
    accs, toks = [], []
    for r in recs:
        cuts = [x["cut"] for x in r["inter"]]
        ans = [x["answer"] for x in r["inter"]]
        # pick the probe closest to the target fraction of segments
        target = frac * r["n_segs"]
        idx = int(np.argmin([abs(c - target) for c in cuts]))
        accs.append(emit_correct(ans[idx], r["gold_raw"], r["gold"]))
        toks.append(cuts[idx] / max(r["n_segs"], 1))
    return np.mean(accs), np.mean(toks)


def eval_adaptive(recs, patience=2):
    accs, toks = [], []
    for r in recs:
        cuts = [x["cut"] for x in r["inter"]]
        ans = [x["answer"] for x in r["inter"]]
        stop_i = len(ans) - 1  # default: last probe
        for i in range(len(ans)):
            if i + 1 < patience:
                continue
            window = ans[i - patience + 1: i + 1]
            if window[0] is not None and all(_eq(window[0], w) for w in window[1:]):
                stop_i = i
                break
        accs.append(emit_correct(ans[stop_i], r["gold_raw"], r["gold"]))
        toks.append(cuts[stop_i] / max(r["n_segs"], 1))
    return np.mean(accs), np.mean(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()

    for tag in args.tags:
        try:
            recs = load(tag)
        except FileNotFoundError:
            continue
        if not recs:
            continue
        fa, ft = eval_full(recs)
        print(f"\n=== {tag} (n={len(recs)}) ===")
        print(f"{'policy':16s} {'acc':>7s} {'tok_frac':>9s} {'acc/full':>9s}")
        print(f"{'full':16s} {fa:7.3f} {ft:9.3f} {1.0:9.3f}")
        for frac in [0.25, 0.5, 0.75]:
            a, t = eval_fixed(recs, frac)
            print(f"{'fixed@%.2f'%frac:16s} {a:7.3f} {t:9.3f} {a/fa if fa else 0:9.3f}")
        for pat in [2, 3]:
            a, t = eval_adaptive(recs, patience=pat)
            print(f"{'ours(pat=%d)'%pat:16s} {a:7.3f} {t:9.3f} {a/fa if fa else 0:9.3f}")


if __name__ == "__main__":
    main()
