"""Regenerate rc_cells.json for the risk-coverage figure: endpoint-only ReCUE,
29 headline cells (no Llama, excluded AIME cell dropped). Per cell stores
y, recue (full endpoint OOF), sc8 (vote fraction), ptrue, selfcert."""
import json, numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from recue.env import EXP_ROOT
from recue import baselines as S

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
out = {}
for Z in [Z1, Z2]:
    for t in sorted({k.split("::")[0] for k in Z.files}):
        ds = t.split("_")[0]
        if ds == "amc23" or t in EXCLUDE or ds not in HEAD:
            continue
        y = Z[f"{t}::y"]
        A = np.hstack([Z[f"{t}::AGREE"], Z[f"{t}::FLL"], Z[f"{t}::FCONF"]]); T = Z[f"{t}::TRACE"]
        recue = oof(np.hstack([A, T]), y)
        pt = Z[f"{t}::PT"].ravel()
        conf = json.loads((EXP_ROOT / "conf" / f"{t}_conf.json").read_text())
        labs = json.loads((EXP_ROOT / "labels" / f"{t}.json").read_text())
        probe = {r["id"] for r in json.loads((EXP_ROOT / "probe" / f"{t}_probe.json").read_text())}
        gen = {r["id"]: r for r in json.loads((EXP_ROOT / "gen" / f"{t}.json").read_text())}
        reb = {r["id"] for r in json.loads((EXP_ROOT / "rebuttal" / f"{t}_reb.json").read_text())}
        saf = EXP_ROOT / "sampans" / f"{t}.json"; sa = json.loads(saf.read_text()) if saf.exists() else {}
        vote, sc_ = [], []
        for r in conf:
            rid = r["id"]
            if not r["intermediate"] or rid not in labs or rid not in probe or rid not in gen or rid not in reb:
                continue
            sc_.append(S.sig_self_certainty(gen[rid]))
            ans = [a for a in sa.get(rid, [])[:8] if a is not None]; c = Counter(ans); tot = sum(c.values())
            vote.append(c.most_common(1)[0][1] / tot if tot else 0.0)
        out[t] = {"y": y.tolist(), "recue": recue.tolist(), "sc8": vote,
                  "selfcert": sc_, "ptrue": pt.tolist() if np.isfinite(pt).all() else None}

json.dump(out, open(f"{EXP_ROOT}/rc_cells.json", "w"))
print(f"saved rc_cells.json with {len(out)} cells (no Llama, endpoint-only)")
