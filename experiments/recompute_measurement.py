"""Recompute app_measurement interface refinements on endpoint-only features,
29 headline cells. Two matched deltas:
 (1) agreement: identity-confidence base -> + primary/re-elicited agreement bit.
 (2) likelihood: first-token confidence base -> length-normalized full-answer LL.
"""
import numpy as np
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
cells = {}
for Z in [Z1, Z2]:
    for t in sorted({k.split("::")[0] for k in Z.files}):
        ds = t.split("_")[0]
        if ds == "amc23" or t in EXCLUDE or ds not in HEAD:
            continue
        cells[t] = Z

# FLL columns: [mean_ll, min_ll, head_ll]; FCONF: [first-token conf]; AGREE: [bit]
def macro(featfn):
    vals = []
    for t, Z in cells.items():
        y = Z[f"{t}::y"]
        vals.append(roc_auc_score(y, oof(featfn(t, Z), y)))
    return float(np.mean(vals))

def col(Z, t, name):
    return Z[f"{t}::{name}"]

# (1) agreement refinement: base = confidence channels (FCONF + FLL), +AGREE bit
base1 = macro(lambda t, Z: np.hstack([col(Z, t, "FCONF"), col(Z, t, "FLL")]))
aug1 = macro(lambda t, Z: np.hstack([col(Z, t, "FCONF"), col(Z, t, "FLL"), col(Z, t, "AGREE")]))
# (2) likelihood refinement: base = first-token conf + agreement; +full-answer LL
base2 = macro(lambda t, Z: np.hstack([col(Z, t, "FCONF"), col(Z, t, "AGREE")]))
aug2 = macro(lambda t, Z: np.hstack([col(Z, t, "FCONF"), col(Z, t, "AGREE"), col(Z, t, "FLL")]))

print(f"(1) agreement:   base {base1:.3f} -> +agree {aug1:.3f}  (delta {aug1-base1:+.3f})")
print(f"(2) likelihood:  base {base2:.3f} -> +full-LL {aug2:.3f}  (delta {aug2-base2:+.3f})")
