"""Build all numbers for the PIVOTED paper from the dumped npz + gen/sampans/ptrue.

New method = two orthogonal single-trace views (naming TBD):
  ACTIVE  re-commitment : AGREE + FLL(3) + FCONF + CONVP           (strong endpoint, 0.850)
  PASSIVE trace profile : TRACE (UTP token-uncertainty trajectory)  (0.802)
  FULL                  : ACTIVE + PASSIVE                          (0.896)

Outputs per-dataset (5) + macro AUROC and AURC + excess-AURC for every main-table row,
plus risk-coverage curves for 3 representative cells, to rebuttal_pivot_tables.json.
Baselines (mean-logprob/self-certainty/deepconf) from gen; SC@8 vote from sampans;
P(True) from ptrue cache. All judge-free, same 5x5 logistic head as the paper.
"""
from __future__ import annotations
import argparse, json
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
    x = np.asarray(x, float)
    if x.ndim == 1: x = x.reshape(-1, 1)
    if not np.isfinite(x).all():
        cmin = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0)
        cmin = np.where(np.isfinite(cmin), cmin, 0.0)
        idx = np.where(~np.isfinite(x)); x[idx] = np.take(cmin, idx[1])
    return x


def oof(X, y):
    X = clean(X); acc = np.zeros(len(y))
    for s in SEEDS:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        acc += o
    return acc / len(SEEDS)


def hb(cell, n=2000, seed=0, metric="auroc"):
    rng = np.random.RandomState(seed); K = len(cell); ds = []
    for _ in range(n):
        cs = rng.choice(K, K, True); d = []
        for ci in cs:
            y, a, b = cell[ci]
            ip = np.where(y == 1)[0]; ng = np.where(y == 0)[0]
            if not len(ip) or not len(ng): continue
            ii = np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ng, len(ng), True)])
            yy = y[ii]
            try:
                if metric == "auroc": d.append(roc_auc_score(yy, a[ii]) - roc_auc_score(yy, b[ii]))
                else: d.append(aurc(b[ii], yy) - aurc(a[ii], yy))
            except Exception: pass
        if d: ds.append(np.mean(d))
    ds = np.array(ds)
    return float(np.mean(ds)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(np.mean(ds <= 0))


def dset(tag): return tag.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()
    Z = np.load(args.npz)
    tags = sorted({k.split("::")[0] for k in Z.files})
    g = lambda t, n: Z[f"{t}::{n}"]

    # scalar baselines + SC vote from raw caches (no _eq needed; vote uses string counts)
    scalar = {}   # tag -> {name: score array}
    for tag in tags:
        gen = {r["id"]: r for r in json.loads((EXP_ROOT / "gen" / f"{tag}.json").read_text())}
        saf = EXP_ROOT / "sampans" / f"{tag}.json"
        sampans = json.loads(saf.read_text()) if saf.exists() else {}
        # rebuild id order exactly as the npz did: iterate conf, keep same filter
        conf = json.loads((EXP_ROOT / "conf" / f"{tag}_conf.json").read_text())
        labs = json.loads((EXP_ROOT / "labels" / f"{tag}.json").read_text())
        probe = {r["id"] for r in json.loads((EXP_ROOT / "probe" / f"{tag}_probe.json").read_text())}
        reb = {r["id"] for r in json.loads((EXP_ROOT / "rebuttal" / f"{tag}_reb.json").read_text())}
        mlp, sc_, dc_, vote, ent = [], [], [], [], []
        for r in conf:
            rid = r["id"]
            if (not r["intermediate"] or rid not in labs or rid not in probe
                    or rid not in gen or rid not in reb):
                continue
            gg = gen[rid]
            mlp.append(S.sig_mean_logprob(gg)); sc_.append(S.sig_self_certainty(gg))
            dc_.append(S.sig_deepconf_bottom(gg))
            ans = [a for a in sampans.get(rid, [])[:8] if a is not None]
            c = Counter(ans); tot = sum(c.values())
            vote.append(c.most_common(1)[0][1] / tot if tot else 0.0)
            ent.append(-sum((v / tot) * np.log(v / tot) for v in c.values()) if tot else 0.0)
        scalar[tag] = {"mean_logprob": np.array(mlp), "self_certainty": np.array(sc_),
                       "deepconf": np.array(dc_), "vote": np.array(vote), "ent": np.array(ent)}

    # row -> per-tag score array
    ACTIVE = ["AGREE", "FLL", "FCONF", "CONVP"]
    scores = defaultdict(dict)
    for tag in tags:
        y = g(tag, "y")
        A = np.hstack([g(tag, b) for b in ACTIVE])
        T = g(tag, "TRACE")
        PT = g(tag, "PT")
        scores["active"][tag] = oof(A, y)
        scores["passive"][tag] = oof(T, y)
        scores["full"][tag] = oof(np.hstack([A, T]), y)
        scores["full+pt"][tag] = oof(np.hstack([A, T, PT]), y)
        scores["multidepth"][tag] = oof(g(tag, "MULTI"), y)
        scores["convergence"][tag] = oof(g(tag, "CONVP"), y)
        # P(True): scalar if present else skip in aggregate
        pt = PT.ravel()
        scores["ptrue"][tag] = pt if np.isfinite(pt).all() else np.full(len(y), np.nan)
        for nm in ("mean_logprob", "self_certainty", "deepconf"):
            scores[nm][tag] = scalar[tag][nm]
        scores["sc8"][tag] = scalar[tag]["vote"]
        v = scalar[tag]["vote"].reshape(-1, 1); e = scalar[tag]["ent"].reshape(-1, 1)
        scores["full+sc8"][tag] = oof(np.hstack([A, T, v, e]), y)

    # aggregate per-dataset + macro AUROC/AURC/excess-AURC
    def agg(row):
        by_ds = defaultdict(list); macro_au, macro_rc, macro_ex = [], [], []
        for tag in tags:
            s = scores[row][tag]
            if not np.isfinite(s).all(): continue
            y = g(tag, "y")
            au = roc_auc_score(y, s); rc = aurc(s, y); acc = y.mean()
            ex = rc / (1 - acc) if acc < 1 else np.nan
            by_ds[dset(tag)].append(au); macro_au.append(au); macro_rc.append(rc)
            if np.isfinite(ex): macro_ex.append(ex)
        return ({d: float(np.mean(v)) for d, v in by_ds.items()},
                float(np.mean(macro_au)) if macro_au else None,
                float(np.mean(macro_rc)) if macro_rc else None,
                float(np.mean(macro_ex)) if macro_ex else None,
                len(macro_au))

    ROWS = ["mean_logprob", "self_certainty", "deepconf", "ptrue", "convergence",
            "multidepth", "active", "passive", "full", "full+pt", "sc8", "full+sc8"]
    print(f"{'row':16s}{'GSM8K':>7s}{'MATH':>7s}{'Miner':>7s}{'Olymp':>7s}{'AMC':>7s}"
          f"{'MACRO':>7s}{'AURC':>7s}{'exAURC':>8s}{'n':>4s}")
    print("-" * 82)
    table = {}
    dmap = {"gsm8k": "GSM8K", "math500": "MATH", "minerva": "Miner", "olympiad": "Olymp", "amc23": "AMC"}
    for row in ROWS:
        by_ds, mau, mrc, mex, n = agg(row)
        table[row] = {"by_dataset": by_ds, "macro_auroc": mau, "macro_aurc": mrc,
                      "excess_aurc": mex, "n": n}
        cells = "".join(f"{by_ds.get(k, float('nan')):7.3f}" for k in
                        ["gsm8k", "math500", "minerva", "olympiad", "amc23"])
        print(f"{row:16s}{cells}{mau or float('nan'):7.3f}{mrc or float('nan'):7.3f}"
              f"{mex or float('nan'):8.3f}{n:4d}")

    # key contrasts
    print("\n=== contrasts ===")
    cons = {}
    for name, A, B in [("full - passive", "full", "passive"),
                       ("full - active", "full", "active"),
                       ("full - sc8", "full+sc8", "sc8"),
                       ("active - ptrue", "active", "ptrue"),
                       ("full+pt - full", "full+pt", "full")]:
        cell = []
        for t in tags:
            sa, sb = scores[A][t], scores[B][t]
            if np.isfinite(sa).all() and np.isfinite(sb).all():
                cell.append((g(t, "y"), sa, sb))
        m, lo, hi, p = hb(cell)
        wins = sum(roc_auc_score(c[0], c[1]) > roc_auc_score(c[0], c[2]) for c in cell)
        print(f"  {name:16s} Δ{m:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} wins {wins}/{len(cell)}")
        cons[name] = [m, lo, hi, p, wins, len(cell)]

    # risk-coverage curves for 3 representative cells
    rc_curves = {}
    for tag in ["math500_qwen8b_k8", "olympiad_qwen14b_k8", "gsm8k_qwen8b_k8"]:
        if tag not in tags: continue
        y = g(tag, "y"); cur = {}
        covs = np.linspace(0.05, 1.0, 20)
        for row in ["passive", "active", "full", "sc8"]:
            s = scores[row][tag]
            cur[row] = [risk_at_coverage(s, y, c) for c in covs]
        rc_curves[tag] = {"coverage": covs.tolist(), "risk": cur, "acc": float(y.mean())}

    json.dump({"table": table, "contrasts": cons, "rc_curves": rc_curves, "n_cells": len(tags)},
              open(args.npz.replace(".npz", "_pivot.json"), "w"), indent=2)
    print("saved", args.npz.replace(".npz", "_pivot.json"))


if __name__ == "__main__":
    main()
