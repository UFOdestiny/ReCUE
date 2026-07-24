"""Per-dataset AURC + macro excess-AURC + macro risk@{10,20,50}% coverage for the
main-table rows, from ladder_feats.npz (+ scalar/vote/ptrue caches). Caches OOF
scores to ladder_scores.npz for reuse."""
import json, os
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


Z = np.load(f"{EXP_ROOT}/ladder_feats.npz")
tags = sorted({k.split("::")[0] for k in Z.files})
g = lambda t, n: Z[f"{t}::{n}"]
dset = lambda t: t.split("_")[0]

scores = defaultdict(dict); ys = {}
for tag in tags:
    y = g(tag, "y"); ys[tag] = y
    A = np.hstack([g(tag, b) for b in ["AGREE", "FLL", "FCONF", "CONVP"]]); T = g(tag, "TRACE")
    scores["active"][tag] = oof(A, y)
    scores["passive"][tag] = oof(T, y)
    scores["full"][tag] = oof(np.hstack([A, T]), y)
    pt = g(tag, "PT").ravel()
    scores["ptrue"][tag] = pt if np.isfinite(pt).all() else None
    gen = {r["id"]: r for r in json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())}
    saf = EXP_ROOT / "sampans" / f"{tag}.json"; sa = json.loads(saf.read_text()) if saf.exists() else {}
    conf = json.loads((EXP_ROOT / "conf" / f"{tag}_conf.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{tag}.json").read_text())
    probe = {r["id"] for r in json.loads((EXP_ROOT / "probe" / f"{tag}_probe.json").read_text())}
    reb = {r["id"] for r in json.loads((EXP_ROOT / "rebuttal" / f"{tag}_reb.json").read_text())}
    sc_, vote, ent = [], [], []
    for r in conf:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in probe or rid not in gen or rid not in reb:
            continue
        sc_.append(S.sig_self_certainty(gen[rid]))
        ans = [a for a in sa.get(rid, [])[:8] if a is not None]; c = Counter(ans); tot = sum(c.values())
        vote.append(c.most_common(1)[0][1] / tot if tot else 0.0)
        ent.append(-sum((v / tot) * np.log(v / tot) for v in c.values()) if tot else 0.0)
    scores["self_certainty"][tag] = np.array(sc_)
    scores["sc8"][tag] = np.array(vote)
    v = np.array(vote).reshape(-1, 1); e = np.array(ent).reshape(-1, 1)
    scores["full+sc8"][tag] = oof(np.hstack([A, T, v, e]), y)

ROWS = ["self_certainty", "ptrue", "passive", "active", "full", "sc8", "full+sc8"]
DS = ["gsm8k", "math500", "minerva", "olympiad", "amc23"]
out = {}
print(f"{'row':14s}" + "".join(f"{d[:5]:>7s}" for d in DS) + f"{'mAURC':>7s}{'exAURC':>8s}{'r@10':>7s}{'r@20':>7s}{'r@50':>7s}")
for row in ROWS:
    bd = defaultdict(list); mac, ex, r10, r20, r50 = [], [], [], [], []
    for t in tags:
        s = scores[row].get(t)
        if s is None: continue
        y = ys[t]; rc = aurc(s, y); acc = y.mean()
        bd[dset(t)].append(rc); mac.append(rc)
        if acc < 1: ex.append(rc / (1 - acc))
        r10.append(risk_at_coverage(s, y, 0.10)); r20.append(risk_at_coverage(s, y, 0.20)); r50.append(risk_at_coverage(s, y, 0.50))
    cells = "".join(f"{np.mean(bd[d]):7.3f}" if bd[d] else f"{'--':>7s}" for d in DS)
    print(f"{row:14s}{cells}{np.mean(mac):7.3f}{np.mean(ex):8.3f}{np.mean(r10):7.3f}{np.mean(r20):7.3f}{np.mean(r50):7.3f}")
    out[row] = {"by_dataset_aurc": {d: float(np.mean(bd[d])) for d in DS if bd[d]},
                "macro_aurc": float(np.mean(mac)), "excess_aurc": float(np.mean(ex)),
                "risk_at": {"10": float(np.mean(r10)), "20": float(np.mean(r20)), "50": float(np.mean(r50))}}
json.dump(out, open(f"{EXP_ROOT}/aurc_table.json", "w"), indent=2)
np.savez(f"{EXP_ROOT}/ladder_scores.npz",
         **{f"{r}::{t}": scores[r][t] for r in ROWS for t in tags if scores[r].get(t) is not None},
         **{f"y::{t}": ys[t] for t in tags})
print("saved aurc_table.json + ladder_scores.npz")
