"""Build the cdyn feature cache ({id: {conv:[6], cdyn:[6]}}) from conf caches.

conv  = answer-identity convergence features  [agree, last_half, final_stable_run,
        conv_frac, flip_rate, n_distinct]
cdyn  = answer-CONFIDENCE dynamics features    [mean_lp, last_lp, min_lp, slope,
        first-agree-conf, std_lp]

This is exactly the feature construction used inline by experiments.main_comparison
(conv_cdyn), factored out so every cell — including the new non-math cells — has a
reproducible cdyn cache. Idempotent: skips cells whose cache already exists.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from acd.env import EXP_ROOT
from acd.features import _eq


def conv_cdyn(rec):
    inter = rec["intermediate"]; neu = [x["neutral"] for x in inter]
    lp = [x.get("neutral_lp") for x in inter]
    n = len(neu); final = neu[-1] if neu else None; half = n // 2
    agree = [1.0 if (a is not None and final is not None and _eq(a, final)) else 0.0 for a in neu]
    af = np.mean(agree) if n else 0.0
    lh = np.mean(agree[half:]) if n - half > 0 else af
    run = 0
    for i in range(n - 1, -1, -1):
        if agree[i] == 1.0:
            run += 1
        else:
            break
    fst = run / n if n else 0.0
    conv = 1.0
    for i in range(n):
        if all(agree[j] == 1.0 for j in range(i, n)):
            conv = (i + 1) / n; break
    ids, reps = [], []
    for a in neu:
        if a is None:
            ids.append(-1); continue
        f = next((k for k, r in enumerate(reps) if _eq(a, r)), None)
        if f is None:
            reps.append(a); f = len(reps) - 1
        ids.append(f)
    flip = sum(1 for i in range(1, n) if ids[i] != ids[i - 1]) / max(1, n - 1)
    CONV = [af, lh, fst, conv, flip, len(set(i for i in ids if i != -1))]
    lpv = [v for v in lp if v is not None]
    slope = float(np.polyfit(np.arange(len(lpv)), lpv, 1)[0]) if len(lpv) >= 2 else 0.0
    fc = -10.0
    for a, v in zip(neu, lp):
        if a is not None and final is not None and _eq(a, final) and v is not None:
            fc = v; break
    CDYN = [float(np.mean(lpv)) if lpv else -10.0, lpv[-1] if lpv else -10.0,
            float(np.min(lpv)) if lpv else -10.0, slope, fc,
            float(np.std(lpv)) if len(lpv) > 1 else 0.0]
    return CONV, CDYN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None,
                    help="cell tags; default = every conf cache")
    args = ap.parse_args()
    (EXP_ROOT / "cdyn").mkdir(exist_ok=True)
    if args.tags:
        confs = [EXP_ROOT / "conf" / f"{t}_conf.json" for t in args.tags]
    else:
        confs = [__import__("pathlib").Path(p) for p in
                 sorted(glob.glob(str(EXP_ROOT / "conf" / "*_conf.json")))]
    for cf in confs:
        tag = os.path.basename(str(cf))[:-len("_conf.json")]
        out = EXP_ROOT / "cdyn" / f"{tag}.json"
        if out.exists():
            print("skip", tag); continue
        if not cf.exists():
            print("MISSING conf", tag); continue
        recs = json.loads(cf.read_text())
        d = {}
        for r in recs:
            if not r.get("intermediate"):
                continue
            cv, cd = conv_cdyn(r)
            d[r["id"]] = {"conv": cv, "cdyn": cd}
        out.write_text(json.dumps(d))
        print("cdyn", tag, len(d))
    print("CDYN_CACHE_DONE")


if __name__ == "__main__":
    main()
