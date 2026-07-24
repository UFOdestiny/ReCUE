"""Exp3 analysis: teacher-forced original-answer support baseline vs ARC.
Builds support features [mean_ll, min_ll, first_lp] from tforce/{tag}_tf.json and
compares, under the same logistic head/folds/labels (endpoint-only, 29 cells):
  TUP, Original-answer support, TUP+support, ARC, ARC+TUP(=ReCUE), ReCUE+support.
Writes teacherforce_table.json.
"""
import json, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from recue.env import EXP_ROOT

SEEDS = [2026, 7, 13, 42, 100]
EXCLUDE = {"aime_qwen35_9b_k8", "math500_llama8b_k8"}
HEAD = ["gsm8k", "math500", "minerva", "olympiad", "aime"]


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


Z1 = np.load(f"{EXP_ROOT}/ladder_feats.npz"); Z2 = np.load(f"{EXP_ROOT}/aime_feats.npz")
rows = defaultdict(list)
missing = []
for Z in [Z1, Z2]:
    for t in sorted({k.split("::")[0] for k in Z.files}):
        ds = t.split("_")[0]
        if ds == "amc23" or t in EXCLUDE or ds not in HEAD:
            continue
        tf_path = EXP_ROOT / "tforce" / f"{t}_tf.json"
        if not tf_path.exists():
            missing.append(t); continue
        tf = json.loads(tf_path.read_text())
        # align to the order used in feature dump: iterate conf like the dump does
        conf = json.loads((EXP_ROOT / "conf" / f"{t}_conf.json").read_text())
        labs = json.loads((EXP_ROOT / "labels" / f"{t}.json").read_text())
        probe = {r["id"] for r in json.loads((EXP_ROOT / "probe" / f"{t}_probe.json").read_text())}
        gen = {r["id"] for r in json.loads((EXP_ROOT / "gen" / f"{t}.json").read_text())}
        reb = {r["id"] for r in json.loads((EXP_ROOT / "rebuttal" / f"{t}_reb.json").read_text())}
        y = Z[f"{t}::y"]
        ARC = np.hstack([Z[f"{t}::AGREE"], Z[f"{t}::FLL"], Z[f"{t}::FCONF"]])
        TUP = Z[f"{t}::TRACE"]
        supp = []
        idx = 0; keep = []
        for r in conf:
            rid = r["id"]
            if not r["intermediate"] or rid not in labs or rid not in probe or rid not in gen or rid not in reb:
                continue
            rec = tf.get(rid)
            if rec is None:
                supp.append([-10.0, -10.0, -10.0])
            else:
                tl = rec["tok_lps"]
                supp.append([float(np.mean(tl)), float(np.min(tl)), float(rec["first_lp"])])
            idx += 1
        supp = np.array(supp)
        if len(supp) != len(y):
            missing.append(f"{t}(len {len(supp)}!={len(y)})"); continue
        rows["_cells"].append(t)
        rows["y"].append(y); rows["ARC"].append(ARC); rows["TUP"].append(TUP); rows["SUP"].append(supp)

cells = rows["_cells"]
print(f"cells with teacher-forcing: {len(cells)}; missing: {len(missing)}")
if missing:
    print("  missing:", missing)


def macro(featfn):
    vals = []
    for i in range(len(cells)):
        y = rows["y"][i]
        s = oof(featfn(i), y)
        vals.append(roc_auc_score(y, s))
    return float(np.mean(vals)), len(cells)


defs = {
    "TUP": lambda i: rows["TUP"][i],
    "Original-answer support": lambda i: rows["SUP"][i],
    "TUP + support": lambda i: np.hstack([rows["TUP"][i], rows["SUP"][i]]),
    "ARC": lambda i: rows["ARC"][i],
    "ARC + TUP (ReCUE)": lambda i: np.hstack([rows["ARC"][i], rows["TUP"][i]]),
    "ReCUE + support": lambda i: np.hstack([rows["ARC"][i], rows["TUP"][i], rows["SUP"][i]]),
    "support + ARC (no TUP)": lambda i: np.hstack([rows["SUP"][i], rows["ARC"][i]]),
}
out = {}
print(f"\n{'representation':28s}{'AUROC':>8s}")
for name, fn in defs.items():
    au, n = macro(fn)
    print(f"{name:28s}{au:8.3f}")
    out[name] = {"auroc": au, "n_cells": n}
json.dump(out, open(f"{EXP_ROOT}/teacherforce_table.json", "w"), indent=2)
print("\nsaved teacherforce_table.json")
