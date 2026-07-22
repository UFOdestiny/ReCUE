"""Mechanism visualization for the answer-confidence-dynamics novelty.

Figures (saved to EXP_ROOT/figs):
  1. Mean forced-answer logprob vs normalized reasoning position, split correct/wrong.
  2. Agreement-with-final vs position (answer-convergence view), correct/wrong.
  3. Distribution of key features (lp_slope, last_lp, final_stable_run) by correctness.
Shows correct traces commit to a HIGH-confidence answer and rise; wrong traces stay
low/volatile — a mechanism distinct from mere answer identity convergence.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from acd.env import EXP_ROOT
from acd.features import _eq

FIGS = EXP_ROOT / "figs"
FIGS.mkdir(exist_ok=True)


def resample(vals, grid=10):
    """Resample a per-cut sequence onto a fixed [0,1] grid of length `grid`."""
    vals = [v for v in vals if v is not None]
    if len(vals) == 0:
        return np.full(grid, np.nan)
    if len(vals) == 1:
        return np.full(grid, vals[0])
    xp = np.linspace(0, 1, len(vals))
    xg = np.linspace(0, 1, grid)
    return np.interp(xg, xp, vals)


def load(tag):
    recs = json.loads((EXP_ROOT / "conf" / f"{tag}_conf.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{tag}.json").read_text())
    return recs, labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--grid", type=int, default=10)
    args = ap.parse_args()

    lp_c, lp_w, ag_c, ag_w = [], [], [], []
    slope_c, slope_w, last_c, last_w = [], [], [], []
    for tag in args.tags:
        try:
            recs, labs = load(tag)
        except FileNotFoundError:
            continue
        for r in recs:
            inter = r["intermediate"]
            if not inter or r["id"] not in labs:
                continue
            y = labs[r["id"]]
            neu = [x["neutral"] for x in inter]
            lp = [x.get("neutral_lp") for x in inter]
            final = neu[-1] if neu else None
            agree = [1.0 if (a is not None and final is not None and _eq(a, final)) else 0.0 for a in neu]
            lp_grid = resample(lp, args.grid)
            ag_grid = resample(agree, args.grid)
            lpv = [v for v in lp if v is not None]
            slope = np.polyfit(np.arange(len(lpv)), lpv, 1)[0] if len(lpv) >= 2 else 0.0
            last = lpv[-1] if lpv else np.nan
            if y == 1:
                lp_c.append(lp_grid); ag_c.append(ag_grid); slope_c.append(slope); last_c.append(last)
            else:
                lp_w.append(lp_grid); ag_w.append(ag_grid); slope_w.append(slope); last_w.append(last)

    xg = np.linspace(0, 1, args.grid)
    lp_c, lp_w = np.array(lp_c), np.array(lp_w)
    ag_c, ag_w = np.array(ag_c), np.array(ag_w)

    # Fig 1: confidence trajectory
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for arr, lab, col in [(lp_c, "correct", "#2a7"), (lp_w, "wrong", "#c33")]:
        m = np.nanmean(arr, 0); se = np.nanstd(arr, 0) / np.sqrt(max(len(arr), 1))
        ax[0].plot(xg, m, label=f"{lab} (n={len(arr)})", color=col, lw=2)
        ax[0].fill_between(xg, m - se, m + se, color=col, alpha=0.2)
    ax[0].set_title("Forced-answer confidence (logprob) vs reasoning position")
    ax[0].set_xlabel("normalized reasoning position"); ax[0].set_ylabel("mean forced-answer logprob")
    ax[0].legend()
    for arr, lab, col in [(ag_c, "correct", "#2a7"), (ag_w, "wrong", "#c33")]:
        m = np.nanmean(arr, 0)
        ax[1].plot(xg, m, label=lab, color=col, lw=2)
    ax[1].set_title("Agreement with final answer (answer-convergence)")
    ax[1].set_xlabel("normalized reasoning position"); ax[1].set_ylabel("P(intermediate == final)")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(FIGS / "mechanism_trajectories.png", dpi=130)
    print("saved", FIGS / "mechanism_trajectories.png")

    # Fig 2: feature distributions
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist([np.array(slope_c), np.array(slope_w)], bins=30, label=["correct", "wrong"],
               color=["#2a7", "#c33"], density=True)
    ax[0].set_title("confidence slope (lp trend)"); ax[0].legend(); ax[0].set_xlabel("slope")
    lc = np.array(last_c); lw = np.array(last_w)
    lc = lc[np.isfinite(lc)]; lw = lw[np.isfinite(lw)]
    ax[1].hist([lc, lw], bins=30, label=["correct", "wrong"], color=["#2a7", "#c33"], density=True)
    ax[1].set_title("final forced-answer logprob"); ax[1].legend(); ax[1].set_xlabel("last lp")
    fig.tight_layout(); fig.savefig(FIGS / "mechanism_features.png", dpi=130)
    print("saved", FIGS / "mechanism_features.png")

    # numeric summary
    print(f"\ncorrect: last_lp mean={np.nanmean(last_c):.3f}  slope mean={np.nanmean(slope_c):+.4f}")
    print(f"wrong:   last_lp mean={np.nanmean(last_w):.3f}  slope mean={np.nanmean(slope_w):+.4f}")


if __name__ == "__main__":
    main()
