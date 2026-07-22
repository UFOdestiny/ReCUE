"""Answer-stabilization dynamics features from a single reasoning chain.

Given the ordered intermediate answers (from acd.probe) plus the final answer,
compute judge-free confidence features that summarize the SHAPE of the
intermediate-answer trajectory:

  agree_frac      : fraction of intermediate answers equivalent to final
  last_half_agree : agreement in the 2nd half of the trajectory
  conv_frac       : earliest fraction of the chain after which the answer is
                    stable & equal to final (1.0 = never stabilizes early)
  flip_rate       : normalized count of consecutive-answer changes
  n_distinct      : number of distinct intermediate answers (incl. None handling)
  inter_entropy   : entropy over distinct intermediate answers
  none_frac       : fraction of probes that failed to yield an answer

Convention: build a single scalar "confidence" where higher => more likely
correct, so AUROC is direct. We also expose individual features for a
logistic-regression combiner (still judge-free; trained on correctness labels).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import List, Optional, Dict, Any

import numpy as np

from acd import data as dv


# ---- equivalence with caching ----
_eqcache: Dict[tuple, bool] = {}


def _eq(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return a is b
    if a == b:
        return True
    key = (a, b) if a <= b else (b, a)
    if key in _eqcache:
        return _eqcache[key]
    mv = dv._get_mv()
    val = False
    if mv:
        _p, _v = mv
        try:
            val = bool(_v(_p("$" + a + "$"), _p("$" + b + "$")))
        except Exception:
            val = False
    _eqcache[key] = val
    return val


def _canon(answers: List[Optional[str]]) -> List[int]:
    """Map answers to cluster ids by math-equivalence (None -> -1)."""
    ids: List[int] = []
    reps: List[str] = []
    for a in answers:
        if a is None:
            ids.append(-1)
            continue
        found = None
        for ci, rep in enumerate(reps):
            if _eq(a, rep):
                found = ci
                break
        if found is None:
            reps.append(a)
            found = len(reps) - 1
        ids.append(found)
    return ids


def features(record: Dict[str, Any]) -> Dict[str, float]:
    inter = record.get("intermediate", [])
    answers = [x["answer"] for x in inter]
    cuts = [x["cut"] for x in inter]
    final = record.get("final_answer")
    n = len(answers)
    if n == 0:
        return {"agree_frac": 0.0, "last_half_agree": 0.0, "conv_frac": 1.0,
                "flip_rate": 1.0, "n_distinct": 9.0, "inter_entropy": 3.0,
                "none_frac": 1.0, "final_stable_run": 0.0}

    # cluster including final answer as reference
    all_ans = answers + [final]
    ids = _canon(all_ans)
    inter_ids = ids[:-1]
    final_id = ids[-1]

    agree = [1.0 if (iid == final_id and final_id != -1) else 0.0 for iid in inter_ids]
    agree_frac = float(np.mean(agree))
    half = n // 2
    last_half_agree = float(np.mean(agree[half:])) if n - half > 0 else agree_frac

    # convergence: earliest index i s.t. answers[i:] all == final
    conv_frac = 1.0
    max_cut = max(cuts) if cuts else 1
    for i in range(n):
        if all(agree[j] == 1.0 for j in range(i, n)):
            conv_frac = cuts[i] / max_cut if max_cut else 1.0
            break

    # flip rate over intermediate answers (ignoring None as break)
    flips = sum(1 for i in range(1, n) if inter_ids[i] != inter_ids[i - 1])
    flip_rate = flips / max(1, n - 1)

    distinct = set(x for x in inter_ids if x != -1)
    n_distinct = float(len(distinct)) if distinct else 1.0
    none_frac = float(np.mean([1.0 if x == -1 else 0.0 for x in inter_ids]))

    # entropy over intermediate answer clusters
    from collections import Counter
    c = Counter([x for x in inter_ids if x != -1])
    tot = sum(c.values())
    if tot > 0:
        inter_entropy = -sum((v / tot) * math.log(v / tot) for v in c.values())
    else:
        inter_entropy = 3.0

    # length of the final stable run (from the end, consecutive == final)
    run = 0
    for i in range(n - 1, -1, -1):
        if agree[i] == 1.0:
            run += 1
        else:
            break
    final_stable_run = run / n

    return {"agree_frac": agree_frac, "last_half_agree": last_half_agree,
            "conv_frac": conv_frac, "flip_rate": flip_rate,
            "n_distinct": n_distinct, "inter_entropy": inter_entropy,
            "none_frac": none_frac, "final_stable_run": final_stable_run}


def scalar_confidence(record: Dict[str, Any]) -> float:
    """Unsupervised single scalar: high => confident.

    Simple interpretable combination: mostly last-half agreement + stability,
    penalized by flips and entropy. (No labels used.)
    """
    f = features(record)
    return (0.5 * f["last_half_agree"] + 0.3 * f["final_stable_run"]
            + 0.2 * f["agree_frac"] - 0.3 * f["flip_rate"] - 0.1 * f["inter_entropy"])
