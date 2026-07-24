"""Endpoint-only (CONVP dropped) full table dump for headline 30 cells.
Emits per-dataset AURC, macro AURC, excess AURC, risk@{10,20,50} for every row,
plus per-cell AUROC matrix for active/full/sc8. Writes endpoint_tables.json.
"""
import json
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from recue.env import EXP_ROOT
from recue import baselines as S
from recue.metrics import aurc, risk_at_coverage

SEEDS = [2026, 7, 13, 42, 100]
HEADLINE_DS = ["gsm8k", "math500", "minerva", "olympiad", "aime"]
EXCLUDE = {"aime_qwen35_9b_k8", "math500_llama8b_k8"}   # drop Llama entirely (MATH500-only)


def clean(x):
    x = np.asarray(x, float); x = x.reshape(len(x), -1)
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


Z1 = np.load(f"{EXP_ROOT}/ladder_feats.npz")
Z2 = np.load(f"{EXP_ROOT}/aime_feats.npz")
tags1 = sorted({k.split("::")[0] for k in Z1.files})
tags2 = sorted({k.split("::")[0] for k in Z2.files})

cells = {}
for Z, tags in [(Z1, tags1), (Z2, tags2)]:
    for t in tags:
        ds = t.split("_")[0]
        if ds == "amc23" or t in EXCLUDE or ds not in HEADLINE_DS:
            continue
        cells[t] = {n: Z[f"{t}::{n}"] for n in ["y", "AGREE", "FLL", "FCONF", "TRACE", "PT"]}

scores = defaultdict(dict); ys = {}
for t, B in cells.items():
    y = B["y"]; ys[t] = y
    A = np.hstack([B["AGREE"], B["FLL"], B["FCONF"]]); T = B["TRACE"]   # endpoint-only
    scores["active"][t] = oof(A, y)
    scores["passive"][t] = oof(T, y)
    scores["full"][t] = oof(np.hstack([A, T]), y)
    pt = B["PT"].ravel(); scores["ptrue"][t] = pt if np.isfinite(pt).all() else None
    gen = {r["id"]: r for r in json.loads((EXP_ROOT / "gen" / f"{t}.json").read_text())}
    saf = EXP_ROOT / "sampans" / f"{t}.json"; sa = json.loads(saf.read_text()) if saf.exists() else {}
    conf = json.loads((EXP_ROOT / "conf" / f"{t}_conf.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{t}.json").read_text())
    probe = {r["id"] for r in json.loads((EXP_ROOT / "probe" / f"{t}_probe.json").read_text())}
    reb = {r["id"] for r in json.loads((EXP_ROOT / "rebuttal" / f"{t}_reb.json").read_text())}
    sc_, vote, ent = [], [], []
    for r in conf:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in probe or rid not in gen or rid not in reb:
            continue
        sc_.append(S.sig_self_certainty(gen[rid]))
        ans = [a for a in sa.get(rid, [])[:8] if a is not None]; c = Counter(ans); tot = sum(c.values())
        vote.append(c.most_common(1)[0][1] / tot if tot else 0.0)
        ent.append(-sum((v / tot) * np.log(v / tot) for v in c.values()) if tot else 0.0)
    scores["self_certainty"][t] = np.array(sc_)
    scores["sc8"][t] = np.array(vote)
    v = np.array(vote).reshape(-1, 1); e = np.array(ent).reshape(-1, 1)
    scores["full+sc8"][t] = oof(np.hstack([A, T, v, e]), y)

ROWS = ["self_certainty", "ptrue", "passive", "active", "full", "sc8", "full+sc8"]
out = {}
print(f"{'row':14s}" + "".join(f"{d[:5]:>7s}" for d in HEADLINE_DS) + f"{'mAURC':>7s}{'exAURC':>8s}{'r@10':>7s}{'r@20':>7s}{'r@50':>7s}")
for row in ROWS:
    bd = defaultdict(list); mac, ex, r10, r20, r50 = [], [], [], [], []
    for t in cells:
        s = scores[row].get(t)
        if s is None or not np.isfinite(s).all(): continue
        y = ys[t]; rc = aurc(s, y); acc = y.mean()
        bd[t.split('_')[0]].append(rc); mac.append(rc)
        if acc < 1: ex.append(rc / (1 - acc))
        r10.append(risk_at_coverage(s, y, 0.10)); r20.append(risk_at_coverage(s, y, 0.20)); r50.append(risk_at_coverage(s, y, 0.50))
    cs = "".join(f"{np.mean(bd[d]):7.3f}" if bd[d] else f"{'--':>7s}" for d in HEADLINE_DS)
    print(f"{row:14s}{cs}{np.mean(mac):7.3f}{np.mean(ex):8.3f}{np.mean(r10):7.3f}{np.mean(r20):7.3f}{np.mean(r50):7.3f}")
    out[row] = {"by_dataset_aurc": {d: float(np.mean(bd[d])) for d in HEADLINE_DS if bd[d]},
                "macro_aurc": float(np.mean(mac)), "excess_aurc": float(np.mean(ex)),
                "risk_at": {"10": float(np.mean(r10)), "20": float(np.mean(r20)), "50": float(np.mean(r50))}}

# per-cell AUROC matrix for active / full / sc8 (for app_matrix.tex)
print("\n=== per-cell AUROC (active / full / sc8) ===")
mat = {}
for t in sorted(cells):
    a = roc_auc_score(ys[t], scores["active"][t])
    f = roc_auc_score(ys[t], scores["full"][t])
    sc = roc_auc_score(ys[t], scores["sc8"][t])
    acc = ys[t].mean(); n = len(ys[t])
    mat[t] = {"n": int(n), "acc": float(acc), "active": float(a), "full": float(f), "sc8": float(sc)}
    print(f"  {t:26s} n={n:4d} acc={acc:.3f}  ARC={a:.3f}  full={f:.3f}  sc8={sc:.3f}")
out["per_cell"] = mat
json.dump(out, open(f"{EXP_ROOT}/endpoint_tables.json", "w"), indent=2)
print("\nsaved endpoint_tables.json")
