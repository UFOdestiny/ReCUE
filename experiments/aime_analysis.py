"""AIME decision analysis: does AIME support ReCUE? Per-cell + macro over the
balanced cells (>=10 in each class), with the unstable cell shown but flagged.

Rows: mean_logprob, self_certainty, ptrue, ARC(active), TUP(passive), ReCUE(full),
SC@8, ReCUE+SC@8. Metrics: AUROC + AURC. Contrasts (hier bootstrap over balanced
cells): full-active, full-passive, ReCUE(1x)-SC8(8x), full+sc8 - sc8, active-ptrue.
"""
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
from recue.metrics import aurc

SEEDS = [2026, 7, 13, 42, 100]
MIN_CLASS = 10   # cells with >=10 in each class count toward the headline macro


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


def hb(cell, n=3000, seed=0):
    rng = np.random.RandomState(seed); ds = []
    for _ in range(n):
        cs = rng.choice(len(cell), len(cell), True); d = []
        for ci in cs:
            y, a, b = cell[ci]
            ip = np.where(y == 1)[0]; ng = np.where(y == 0)[0]
            if not len(ip) or not len(ng): continue
            ii = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ng, len(ng), True)]); yy = y[ii]
            try: d.append(roc_auc_score(yy, a[ii]) - roc_auc_score(yy, b[ii]))
            except Exception: pass
        if d: ds.append(np.mean(d))
    ds = np.array(ds)
    return float(np.mean(ds)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(np.mean(ds <= 0))


Z = np.load(f"{EXP_ROOT}/aime_feats.npz")
tags = sorted({k.split("::")[0] for k in Z.files})
g = lambda t, n: Z[f"{t}::{n}"]

scores = defaultdict(dict); ys = {}; balanced = []
for tag in tags:
    y = g(tag, "y"); ys[tag] = y
    A = np.hstack([g(tag, b) for b in ["AGREE", "FLL", "FCONF", "CONVP"]]); T = g(tag, "TRACE")
    scores["active"][tag] = oof(A, y)
    scores["passive"][tag] = oof(T, y)
    scores["full"][tag] = oof(np.hstack([A, T]), y)
    pt = g(tag, "PT").ravel(); scores["ptrue"][tag] = pt if np.isfinite(pt).all() else None
    gen = {r["id"]: r for r in json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())}
    saf = EXP_ROOT / "sampans" / f"{tag}.json"; sa = json.loads(saf.read_text()) if saf.exists() else {}
    conf = json.loads((EXP_ROOT / "conf" / f"{tag}_conf.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{tag}.json").read_text())
    probe = {r["id"] for r in json.loads((EXP_ROOT / "probe" / f"{tag}_probe.json").read_text())}
    reb = {r["id"] for r in json.loads((EXP_ROOT / "rebuttal" / f"{tag}_reb.json").read_text())}
    mlp, sc_, vote, ent = [], [], [], []
    for r in conf:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in probe or rid not in gen or rid not in reb:
            continue
        mlp.append(S.sig_mean_logprob(gen[rid])); sc_.append(S.sig_self_certainty(gen[rid]))
        ans = [a for a in sa.get(rid, [])[:8] if a is not None]; c = Counter(ans); tot = sum(c.values())
        vote.append(c.most_common(1)[0][1] / tot if tot else 0.0)
        ent.append(-sum((v / tot) * np.log(v / tot) for v in c.values()) if tot else 0.0)
    scores["mean_logprob"][tag] = np.array(mlp); scores["self_certainty"][tag] = np.array(sc_)
    scores["sc8"][tag] = np.array(vote)
    v = np.array(vote).reshape(-1, 1); e = np.array(ent).reshape(-1, 1)
    scores["full+sc8"][tag] = oof(np.hstack([A, T, v, e]), y)
    if int(y.sum()) >= MIN_CLASS and int((1 - y).sum()) >= MIN_CLASS:
        balanced.append(tag)

ROWS = ["mean_logprob", "self_certainty", "ptrue", "active", "passive", "full", "sc8", "full+sc8"]
print(f"balanced cells (>= {MIN_CLASS}/class): {[t.replace('aime_','').replace('_k8','') for t in balanced]}")
print(f"excluded (unstable): {[t.replace('aime_','').replace('_k8','') for t in tags if t not in balanced]}\n")

# per-cell AUROC table (all 6, mark excluded)
print(f"{'cell':12s}" + "".join(f"{r[:8]:>9s}" for r in ROWS))
for tag in tags:
    flag = "" if tag in balanced else "*"
    row = ""
    for r in ROWS:
        s = scores[r].get(tag)
        row += f"{roc_auc_score(ys[tag], s):9.3f}" if s is not None and np.isfinite(s).all() else f"{'--':>9s}"
    print(f"{tag.replace('aime_','').replace('_k8',''):11s}{flag}{row}")

print(f"\n=== MACRO over {len(balanced)} balanced cells ===")
for r in ROWS:
    vals = [roc_auc_score(ys[t], scores[r][t]) for t in balanced if scores[r].get(t) is not None and np.isfinite(scores[r][t]).all()]
    rcs = [aurc(scores[r][t], ys[t]) for t in balanced if scores[r].get(t) is not None and np.isfinite(scores[r][t]).all()]
    print(f"  {r:14s} AUROC={np.mean(vals):.3f}  AURC={np.mean(rcs):.3f}  ({len(vals)} cells)")

print("\n=== contrasts (hier bootstrap, balanced cells) ===")
out = {}
for name, A, B in [("full - passive", "full", "passive"), ("full - active", "full", "active"),
                   ("ReCUE(1x) - SC8(8x)", "full", "sc8"), ("full+sc8 - sc8", "full+sc8", "sc8"),
                   ("active - ptrue", "active", "ptrue")]:
    cell = [(ys[t], scores[A][t], scores[B][t]) for t in balanced
            if scores[A].get(t) is not None and scores[B].get(t) is not None
            and np.isfinite(scores[A][t]).all() and np.isfinite(scores[B][t]).all()]
    m, lo, hi, p = hb(cell)
    wins = sum(roc_auc_score(c[0], c[1]) > roc_auc_score(c[0], c[2]) for c in cell)
    sig = "SIG" if lo > 0 or hi < 0 else "ns"
    print(f"  {name:22s} d{m:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} [{sig}] wins {wins}/{len(cell)}")
    out[name] = [m, lo, hi, p, wins, len(cell)]

json.dump({"balanced": balanced, "excluded": [t for t in tags if t not in balanced],
           "macro_auroc": {r: float(np.mean([roc_auc_score(ys[t], scores[r][t]) for t in balanced if scores[r].get(t) is not None and np.isfinite(scores[r][t]).all()])) for r in ROWS},
           "macro_aurc": {r: float(np.mean([aurc(scores[r][t], ys[t]) for t in balanced if scores[r].get(t) is not None and np.isfinite(scores[r][t]).all()])) for r in ROWS},
           "per_cell_auroc": {t: {r: (float(roc_auc_score(ys[t], scores[r][t])) if scores[r].get(t) is not None and np.isfinite(scores[r][t]).all() else None) for r in ROWS} for t in tags},
           "contrasts": out}, open(f"{EXP_ROOT}/aime_analysis.json", "w"), indent=2)
print("saved aime_analysis.json")
