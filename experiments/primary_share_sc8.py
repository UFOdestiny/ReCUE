"""Reviewer control (Issue 1): SC@8 primary-answer vote share vs. modal vote share.

Modal SC@8:   q_K       = (1/K) max_b sum_k 1[a^(k) == b]     (current headline)
Primary SC@8: q_K(a)    = (1/K)     sum_k 1[a^(k) == a]       (share supporting
                                                              the primary answer)

Both are scored against the SAME primary-answer correctness label used everywhere
else in the paper, over the SAME headline cells (30 pairs, AMC/excluded dropped),
macro-averaged over model-dataset pairs exactly as recompute_headline.py does.
Writes primary_share_sc8.json.
"""
import json
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
from recue.env import EXP_ROOT, normalize_num
from recue.metrics import aurc

HEADLINE_DS = ["gsm8k", "math500", "minerva", "olympiad", "aime"]
EXCLUDE = {"aime_qwen35_9b_k8", "math500_llama8b_k8"}


def norm(a):
    if a is None:
        return None
    n = normalize_num(str(a))
    return n if n is not None else str(a).strip()


# discover the exact headline cells from the feature npz tags
Z1 = np.load(f"{EXP_ROOT}/ladder_feats.npz")
Z2 = np.load(f"{EXP_ROOT}/aime_feats.npz")
tags = sorted({k.split("::")[0] for k in list(Z1.files) + list(Z2.files)})
cells = [t for t in tags
         if t.split("_")[0] in HEADLINE_DS and t.split("_")[0] != "amc23"
         and t not in EXCLUDE]

modal_by, primary_by = defaultdict(list), defaultdict(list)
modal_rc_by, primary_rc_by = defaultdict(list), defaultdict(list)
n_cells = 0
for t in cells:
    conf = json.loads((EXP_ROOT / "conf" / f"{t}_conf.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{t}.json").read_text())
    probe = {r["id"]: r for r in json.loads((EXP_ROOT / "probe" / f"{t}_probe.json").read_text())}
    reb = {r["id"] for r in json.loads((EXP_ROOT / "rebuttal" / f"{t}_reb.json").read_text())}
    gen = {r["id"] for r in json.loads((EXP_ROOT / "gen" / f"{t}.json").read_text())}
    sa = json.loads((EXP_ROOT / "sampans" / f"{t}.json").read_text())

    modal, primary, y = [], [], []
    for r in conf:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in probe or rid not in gen or rid not in reb:
            continue
        ans = [norm(a) for a in sa.get(rid, [])[:8] if a is not None]
        tot = len(ans)
        if not tot:
            continue
        c = Counter(ans)
        pa = norm(probe[rid].get("final_answer"))
        modal.append(c.most_common(1)[0][1] / tot)
        primary.append(c.get(pa, 0) / tot)
        y.append(labs[rid])
    y = np.array(y)
    if len(np.unique(y)) < 2:
        continue
    ds = t.split("_")[0]
    modal_by[ds].append(roc_auc_score(y, modal)); primary_by[ds].append(roc_auc_score(y, primary))
    modal_rc_by[ds].append(aurc(np.array(modal), y)); primary_rc_by[ds].append(aurc(np.array(primary), y))
    n_cells += 1


def macro(byds):
    allv = [v for vs in byds.values() for v in vs]
    return float(np.mean(allv)), {d: float(np.mean(v)) for d, v in byds.items()}


m_auc, m_bd = macro(modal_by)
p_auc, p_bd = macro(primary_by)
m_rc, _ = macro(modal_rc_by)
p_rc, _ = macro(primary_rc_by)

print(f"cells used: {n_cells}")
print(f"{'':22s}" + "".join(f"{d[:5]:>9s}" for d in HEADLINE_DS) + f"{'MACRO':>9s}{'AURC':>8s}")
print(f"{'SC@8 modal (current)':22s}" + "".join(f"{m_bd.get(d, float('nan')):9.3f}" for d in HEADLINE_DS) + f"{m_auc:9.3f}{m_rc:8.3f}")
print(f"{'SC@8 primary-share':22s}" + "".join(f"{p_bd.get(d, float('nan')):9.3f}" for d in HEADLINE_DS) + f"{p_auc:9.3f}{p_rc:8.3f}")

out = {"n_cells": n_cells,
       "modal": {"macro_auroc": m_auc, "macro_aurc": m_rc, "by_dataset": m_bd},
       "primary_share": {"macro_auroc": p_auc, "macro_aurc": p_rc, "by_dataset": p_bd}}
json.dump(out, open(f"{EXP_ROOT}/primary_share_sc8.json", "w"), indent=2)
print("saved primary_share_sc8.json")
