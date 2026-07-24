"""Recompute app_learning (classifier-capacity control) on endpoint-only ARC
features, 29 headline cells. In-domain 5x5-fold AUROC for logistic/RF/GBT/MLP,
plus RF's LODO transfer gain over logistic."""
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
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


def mk(kind):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    if kind == "gbt":
        return GradientBoostingClassifier(random_state=0)
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=0))


def oof_auc(X, y, kind):
    X = clean(X); a = np.zeros(len(y))
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = mk(kind).fit(X[tr], y[tr]); o[te] = c.predict_proba(X[te])[:, 1]
        a += o
    return roc_auc_score(y, a / len(SEEDS))


Z1 = np.load(f"{EXP_ROOT}/ladder_feats.npz"); Z2 = np.load(f"{EXP_ROOT}/aime_feats.npz")
cells = {}
for Z in [Z1, Z2]:
    for t in sorted({k.split("::")[0] for k in Z.files}):
        ds = t.split("_")[0]
        if ds == "amc23" or t in EXCLUDE or ds not in HEAD:
            continue
        y = Z[f"{t}::y"]
        ARC = np.hstack([Z[f"{t}::AGREE"], Z[f"{t}::FLL"], Z[f"{t}::FCONF"]])
        cells[t] = {"y": y, "ARC": ARC, "ds": ds, "model": t.split("_", 1)[1]}

print("=== in-domain AUROC (endpoint-only ARC, 29 cells) ===")
indom = {}
for kind in ["logistic", "rf", "gbt", "mlp"]:
    vals = [oof_auc(c["ARC"], c["y"], kind) for c in cells.values()]
    indom[kind] = float(np.mean(vals))
    print(f"  {kind:10s} {indom[kind]:.3f}")

# RF vs logistic LODO transfer gain
print("\n=== LODO transfer gain RF vs logistic ===")
models = sorted({c["model"] for c in cells.values()})
gains = []
for kind in ["logistic", "rf"]:
    tr = []
    for tgt in HEAD:
        for m in models:
            tc = next((c for c in cells.values() if c["ds"] == tgt and c["model"] == m), None)
            if tc is None: continue
            src = [c for c in cells.values() if c["model"] == m and c["ds"] != tgt]
            if len({c["ds"] for c in src}) < 3: continue
            Xtr = np.vstack([c["ARC"] for c in src]); ytr = np.concatenate([c["y"] for c in src])
            y = tc["y"]
            if y.sum() == 0 or (1 - y).sum() == 0: continue
            clf = mk(kind).fit(clean(Xtr), ytr)
            tr.append((tgt + "|" + m, roc_auc_score(y, clf.predict_proba(clean(tc["ARC"]))[:, 1])))
    gains.append(dict(tr))
lodo_log = np.mean(list(gains[0].values())); lodo_rf = np.mean(list(gains[1].values()))
print(f"  logistic LODO {lodo_log:.3f}")
print(f"  rf       LODO {lodo_rf:.3f}  gain {lodo_rf - lodo_log:+.3f}")
