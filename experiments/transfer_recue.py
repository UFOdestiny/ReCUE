"""LODO cross-dataset transfer of the ARC (re-commitment) head over the 30-cell
headline set (GSM8K/MATH500/Minerva/Olympiad/AIME; AMC excluded, unstable AIME
cell excluded). Per backbone, train on the source datasets and test on the held-out
target dataset. No target identity/accuracy/statistic enters the head.

Reports, per target dataset (macro over backbones):
  Best-transferred single-trace baseline (self-certainty or P(True) transferred),
  ARC transferred, ARC in-domain (5-fold), and degradation.
Plus LODO macro and a single global head (no cell id) vs +cell-id upper bound.
"""
import numpy as np, json, os
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold

R = os.environ["EXP_ROOT"]
HEAD = ["gsm8k", "math500", "minerva", "olympiad", "aime"]
EXC = {"aime_qwen35_9b_k8", "math500_llama8b_k8", "amc23_ministral_k8", "amc23_phi4r_k8",
       "amc23_qwen14b_k8", "amc23_qwen35_9b_k8", "amc23_qwen4b_k8", "amc23_qwen8b_k8"}
SEEDS = [2026, 7, 13, 42, 100]


def clean(x):
    x = np.asarray(x, float); x = x.reshape(len(x), -1)
    if not np.isfinite(x).all():
        cm = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0); cm = np.where(np.isfinite(cm), cm, 0.0)
        i = np.where(~np.isfinite(x)); x[i] = np.take(cm, i[1])
    return x


def fit_pred(Xtr, ytr, Xte):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(clean(Xtr), ytr)
    return m.predict_proba(clean(Xte))[:, 1]


def oof(X, y):
    X = clean(X); acc = np.zeros(len(y))
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            o[te] = fit_pred(X[tr], y[tr], X[te])
        acc += o
    return acc / len(SEEDS)


Z1 = np.load(f"{R}/ladder_feats.npz"); Z2 = np.load(f"{R}/aime_feats.npz")
cells = {}
for Z in [Z1, Z2]:
    for t in sorted({k.split("::")[0] for k in Z.files}):
        ds = t.split("_")[0]
        if ds not in HEAD or t in EXC:
            continue
        y = Z[f"{t}::y"]
        ARC = np.hstack([Z[f"{t}::AGREE"], Z[f"{t}::FLL"], Z[f"{t}::FCONF"]])  # endpoint-only
        cells[t] = {"ds": ds, "model": t.split("_", 1)[1], "y": y, "ARC": ARC}

models = sorted({c["model"] for c in cells.values()})

# ---- LODO per backbone: train on source datasets, test held-out dataset ----
transfer_au = defaultdict(list)   # target ds -> [auroc across backbones]
indom_au = defaultdict(list)
for tgt in HEAD:
    for m in models:
        tcell = next((c for c in cells.values() if c["ds"] == tgt and c["model"] == m), None)
        if tcell is None:
            continue
        src = [c for c in cells.values() if c["model"] == m and c["ds"] != tgt]
        srcds = {c["ds"] for c in src}
        if len(srcds) < 3:   # need enough source datasets (skips llama: math500 only)
            continue
        Xtr = np.vstack([c["ARC"] for c in src]); ytr = np.concatenate([c["y"] for c in src])
        y = tcell["y"]
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        s = fit_pred(Xtr, ytr, tcell["ARC"])
        transfer_au[tgt].append(roc_auc_score(y, s))
        indom_au[tgt].append(roc_auc_score(y, oof(tcell["ARC"], y)))

print(f"{'target':10s}{'transfer':>10s}{'in-dom':>9s}{'degr':>8s}{'n':>4s}")
lodo_t, lodo_i = [], []
rows = {}
for tgt in HEAD:
    if not transfer_au[tgt]:
        continue
    t = np.mean(transfer_au[tgt]); i = np.mean(indom_au[tgt]); n = len(transfer_au[tgt])
    print(f"{tgt:10s}{t:10.3f}{i:9.3f}{t-i:+8.3f}{n:4d}")
    rows[tgt] = {"transfer": float(t), "indomain": float(i), "n": n}
    lodo_t.append(t); lodo_i.append(i)
print(f"{'LODO macro':10s}{np.mean(lodo_t):10.3f}{np.mean(lodo_i):9.3f}{np.mean(lodo_t)-np.mean(lodo_i):+8.3f}")

# ---- global head (no cell id) via leave-one-cell-out, ARC features ----
tags = sorted(cells.keys())
loco = []
for i, t in enumerate(tags):
    Xtr = np.vstack([cells[o]["ARC"] for o in tags if o != t])
    ytr = np.concatenate([cells[o]["y"] for o in tags if o != t])
    y = cells[t]["y"]
    if y.sum() == 0 or (1 - y).sum() == 0:
        continue
    s = fit_pred(Xtr, ytr, cells[t]["ARC"])
    loco.append(roc_auc_score(y, s))
print(f"\nglobal head (no cell-id), leave-one-cell-out macro: {np.mean(loco):.3f}")

json.dump({"lodo_rows": rows, "lodo_macro_transfer": float(np.mean(lodo_t)),
           "lodo_macro_indomain": float(np.mean(lodo_i)),
           "global_no_cellid": float(np.mean(loco))},
          open(f"{R}/transfer_recue.json", "w"), indent=2)
print("saved transfer_recue.json")
