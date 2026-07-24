"""Exp1 analysis: decoding-symmetry matrix.
For each (tag, primary_temp, probe_temp) condition dumped by run_decoding_matrix,
build ARC and ReCUE features from the regenerated primary + probe, and report:
  primary accuracy, agreement rate (overall/correct/wrong),
  ARC AUROC, ReCUE AUROC, and the 2x2 answer-transition matrix
  (primary correct/wrong x re-elicited correct/wrong).
Writes decmatrix_table.json.
"""
import json, glob, os, re
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from recue.env import EXP_ROOT

SEEDS = [2026, 7, 13, 42, 100]
NBIN = 8


def clean(x):
    x = np.asarray(x, float).reshape(len(x), -1)
    if not np.isfinite(x).all():
        cm = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
        i = np.where(~np.isfinite(x)); x[i] = np.take(cm, i[1])
    return x


def oof(X, y):
    X = clean(X); a = np.zeros(len(y))
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        a += o
    return a / len(SEEDS)


def _binned(arr, nbin):
    if len(arr) == 0:
        return [0.0] * nbin
    idx = np.linspace(0, len(arr), nbin + 1).astype(int)
    return [float(np.mean(arr[idx[i]:idx[i + 1]])) if idx[i + 1] > idx[i] else 0.0 for i in range(nbin)]


def _entropy_traj(topk):
    ents = []
    for step in topk:
        if not step:
            continue
        lp = np.array(step, float); p = np.exp(lp); p = p / (p.sum() + 1e-12)
        ents.append(float(-(p * np.log(p + 1e-12)).sum()))
    return np.array(ents)


def _slope_r2(arr):
    if len(arr) < 3:
        return 0.0, 0.0
    x = np.arange(len(arr)); s = np.polyfit(x, arr, 1)[0]
    pred = np.polyval(np.polyfit(x, arr, 1), x)
    ss = 1 - np.sum((arr - pred) ** 2) / (np.sum((arr - arr.mean()) ** 2) + 1e-12)
    return float(s), float(ss)


def traceprofile(chosen_lp, topk_lp):
    lp = np.array(chosen_lp, float)
    ent = _entropy_traj(topk_lp)
    if len(lp) == 0:
        return [0.0] * (2 * NBIN + 11)
    feats = _binned(lp, NBIN) + (_binned(ent, NBIN) if len(ent) else [0.0] * NBIN)
    feats += [float(np.mean(lp)), float(np.min(lp)), float(np.std(lp))]
    feats += [float(np.mean(ent)) if len(ent) else 0.0, float(np.max(ent)) if len(ent) else 0.0]
    s_lp, r2_lp = _slope_r2(lp); s_e, r2_e = _slope_r2(ent) if len(ent) else (0.0, 0.0)
    feats += [s_lp, r2_lp, s_e, r2_e, float(np.mean(lp[-30:])), float(np.mean(lp < -2.0))]
    return feats


files = sorted(glob.glob(f"{EXP_ROOT}/decmatrix/*.json"))
out = {}
print(f"{'condition':34s}{'n':>5s}{'acc':>7s}{'agree':>7s}{'ARC':>7s}{'ReCUE':>7s}"
      f"{'cc':>6s}{'cw':>6s}{'wc':>6s}{'ww':>6s}")
for f in files:
    name = os.path.basename(f)[:-5]
    recs = json.loads(open(f).read())
    recs = [r for r in recs if r.get("primary_ans")]
    y = np.array([r["primary_correct"] for r in recs])
    if y.sum() < 5 or (1 - y).sum() < 5:
        print(f"{name:34s}{len(recs):5d}  (degenerate labels, skipped)"); continue
    agree = np.array([r["agree"] for r in recs])
    # ARC features: [agree, mean_ans_lp, min_ans_lp, head_ans_lp, first_lp]
    ARC = []
    for r in recs:
        al = r["ans_lps"] or [-10.0]
        ARC.append([r["agree"], float(np.mean(al)), float(np.min(al)),
                    float(np.mean(al[:2])), r["first_lp"] if r["first_lp"] is not None else -10.0])
    ARC = np.array(ARC)
    TUP = np.array([traceprofile(r["chosen_lp"], r["topk_lp"]) for r in recs])
    arc_au = roc_auc_score(y, oof(ARC, y))
    recue_au = roc_auc_score(y, oof(np.hstack([ARC, TUP]), y))
    pc = y; rc = np.array([r["reelicit_correct"] for r in recs])
    cc = int(((pc == 1) & (rc == 1)).sum()); cw = int(((pc == 1) & (rc == 0)).sum())
    wc = int(((pc == 0) & (rc == 1)).sum()); ww = int(((pc == 0) & (rc == 0)).sum())
    n = len(recs)
    ag_c = agree[pc == 1].mean() if (pc == 1).any() else 0.0
    ag_w = agree[pc == 0].mean() if (pc == 0).any() else 0.0
    print(f"{name:34s}{n:5d}{y.mean():7.3f}{agree.mean():7.3f}{arc_au:7.3f}{recue_au:7.3f}"
          f"{cc/n:6.2f}{cw/n:6.2f}{wc/n:6.2f}{ww/n:6.2f}")
    out[name] = {"n": n, "acc": float(y.mean()), "agree": float(agree.mean()),
                 "agree_correct": float(ag_c), "agree_wrong": float(ag_w),
                 "arc_auroc": float(arc_au), "recue_auroc": float(recue_au),
                 "transitions": {"cc": cc, "cw": cw, "wc": wc, "ww": ww}}
json.dump(out, open(f"{EXP_ROOT}/decmatrix_table.json", "w"), indent=2)
print("\nsaved decmatrix_table.json")
