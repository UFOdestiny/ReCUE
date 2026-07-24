"""Recompute all headline numbers under the new benchmark set:
GSM8K, MATH500, Minerva, Olympiad, AIME (AMC23 -> appendix; qwen35_9b AIME excluded).

Merges ladder_feats.npz (31 cells incl amc) + aime_feats.npz (6 aime cells).
Emits per-dataset + macro AUROC & AURC for the main result table, the AURC table,
and the ablation ladder, plus the key contrasts. Writes headline_v2.json.
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
from recue.metrics import aurc, risk_at_coverage

SEEDS = [2026, 7, 13, 42, 100]
HEADLINE_DS = ["gsm8k", "math500", "minerva", "olympiad", "aime"]
EXCLUDE = {"aime_qwen35_9b_k8",      # unstable (6 positives)
           "math500_llama8b_k8"}     # Llama: only MATH500, non-reasoning; dropped entirely


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


def dset(t): return t.split("_")[0].replace("aime", "aime")  # aime_qwen4b -> aime


# ---- load both npz, build unified cell -> feature blocks ----
Z1 = np.load(f"{EXP_ROOT}/ladder_feats.npz")
Z2 = np.load(f"{EXP_ROOT}/aime_feats.npz")
tags1 = sorted({k.split("::")[0] for k in Z1.files})
tags2 = sorted({k.split("::")[0] for k in Z2.files})

def blocks(Z, t, name): return Z[f"{t}::{name}"]

cells = {}   # tag -> dict of feature arrays + y
for Z, tags in [(Z1, tags1), (Z2, tags2)]:
    for t in tags:
        ds = t.split("_")[0]
        if ds == "amc23":       # AMC -> appendix, not headline
            continue
        if t in EXCLUDE:
            continue
        if ds not in HEADLINE_DS:
            continue
        cells[t] = {n: blocks(Z, t, n) for n in ["y", "AGREE", "FLL", "FCONF", "CONVP", "MULTI", "TRACE", "PT"]}

print(f"headline cells: {len(cells)}")
byds = Counter(t.split('_')[0] for t in cells)
print("per-dataset cell counts:", dict(byds))

# scores per row
scores = defaultdict(dict); ys = {}
for t, B in cells.items():
    y = B["y"]; ys[t] = y
    A = np.hstack([B["AGREE"], B["FLL"], B["FCONF"]]); T = B["TRACE"]  # endpoint-only (CONVP dropped)
    scores["active"][t] = oof(A, y)
    scores["passive"][t] = oof(T, y)
    scores["full"][t] = oof(np.hstack([A, T]), y)
    scores["multidepth"][t] = oof(B["MULTI"], y)
    scores["arc+multi"][t] = oof(np.hstack([A, B["MULTI"]]), y)
    scores["full+multi"][t] = oof(np.hstack([A, T, B["MULTI"]]), y)
    pt = B["PT"].ravel(); scores["ptrue"][t] = pt if np.isfinite(pt).all() else None
    gen = {r["id"]: r for r in json.loads((EXP_ROOT / "gen" / f"{t}.json").read_text())}
    saf = EXP_ROOT / "sampans" / f"{t}.json"; sa = json.loads(saf.read_text()) if saf.exists() else {}
    conf = json.loads((EXP_ROOT / "conf" / f"{t}_conf.json").read_text())
    labs = json.loads((EXP_ROOT / "labels" / f"{t}.json").read_text())
    probe = {r["id"] for r in json.loads((EXP_ROOT / "probe" / f"{t}_probe.json").read_text())}
    reb = {r["id"] for r in json.loads((EXP_ROOT / "rebuttal" / f"{t}_reb.json").read_text())}
    mlp, sc_, vote, ent = [], [], [], []
    for r in conf:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in probe or rid not in gen or rid not in reb:
            continue
        mlp.append(S.sig_mean_logprob(gen[rid])); sc_.append(S.sig_self_certainty(gen[rid]))
        ans = [a for a in sa.get(rid, [])[:8] if a is not None]; c = Counter(ans); tot = sum(c.values())
        vote.append(c.most_common(1)[0][1] / tot if tot else 0.0)
        ent.append(-sum((v / tot) * np.log(v / tot) for v in c.values()) if tot else 0.0)
    scores["mean_logprob"][t] = np.array(mlp); scores["self_certainty"][t] = np.array(sc_)
    # deepconf
    dc = []
    for r in conf:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in probe or rid not in gen or rid not in reb: continue
        dc.append(S.sig_deepconf_bottom(gen[rid]))
    scores["deepconf"][t] = np.array(dc)
    scores["sc8"][t] = np.array(vote)
    v = np.array(vote).reshape(-1, 1); e = np.array(ent).reshape(-1, 1)
    scores["full+sc8"][t] = oof(np.hstack([A, T, v, e]), y)

def agg(row, metric="auroc"):
    bd = defaultdict(list); mac = []
    for t in cells:
        s = scores[row].get(t)
        if s is None or not np.isfinite(s).all(): continue
        y = ys[t]
        val = roc_auc_score(y, s) if metric == "auroc" else aurc(s, y)
        bd[t.split('_')[0]].append(val); mac.append(val)
    return {d: float(np.mean(v)) for d, v in bd.items()}, (float(np.mean(mac)) if mac else None), len(mac)

ROWS = ["mean_logprob", "self_certainty", "deepconf", "ptrue", "multidepth",
        "active", "passive", "full", "arc+multi", "full+multi", "sc8", "full+sc8"]
print(f"\n{'row':14s}" + "".join(f"{d[:5]:>8s}" for d in HEADLINE_DS) + f"{'MACRO':>8s}{'AURC':>7s}{'n':>4s}")
table = {}
for r in ROWS:
    bd, mac, n = agg(r, "auroc")
    _, mrc, _ = agg(r, "aurc")
    cells_str = "".join(f"{bd.get(d, float('nan')):8.3f}" for d in HEADLINE_DS)
    print(f"{r:14s}{cells_str}{mac or float('nan'):8.3f}{mrc or float('nan'):7.3f}{n:4d}")
    table[r] = {"by_dataset_auroc": bd, "macro_auroc": mac, "macro_aurc": mrc, "n": n}

print("\n=== contrasts (hier bootstrap, new headline cells) ===")
cons = {}
for name, A, B in [("full - passive", "full", "passive"), ("full - active", "full", "active"),
                   ("ReCUE(1x) - SC8(8x)", "full", "sc8"), ("full+sc8 - sc8", "full+sc8", "sc8"),
                   ("arc+multi - active", "arc+multi", "active"), ("full+multi - full", "full+multi", "full"),
                   ("active - ptrue", "active", "ptrue")]:
    cl = [(ys[t], scores[A][t], scores[B][t]) for t in cells
          if scores[A].get(t) is not None and scores[B].get(t) is not None
          and np.isfinite(scores[A][t]).all() and np.isfinite(scores[B][t]).all()]
    m, lo, hi, p = hb(cl)
    wins = sum(roc_auc_score(c[0], c[1]) > roc_auc_score(c[0], c[2]) for c in cl)
    sig = "SIG" if lo > 0 or hi < 0 else "ns"
    print(f"  {name:22s} d{m:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} [{sig}] wins {wins}/{len(cl)}")
    cons[name] = [m, lo, hi, p, wins, len(cl)]

json.dump({"headline_ds": HEADLINE_DS, "n_cells": len(cells),
           "per_dataset_cell_counts": dict(byds), "table": table, "contrasts": cons},
          open(f"{EXP_ROOT}/headline_v2.json", "w"), indent=2)
print("saved headline_v2.json")
