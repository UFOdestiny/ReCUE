"""P0-6 statistical protocol upgrade (More_EXP.md Sec 2.6).

Pre-registered PRIMARY contrasts (fixed before looking at results):
  C1  ChainUQ            vs strongest comparable 1x single-trace baseline
                         (max over {mean_logprob, self_certainty, p_true} per cell)
  C2  ordered trajectory vs CONV+FINAL   (endpoint null)
  C3  ChainUQ + SC@8     vs SC@8         (matched-budget fusion)

For every primary contrast we report:
  * per-cell paired problem-level bootstrap Δ + 95% CI + p
  * Holm-Bonferroni correction over the 31 cells (also uncorrected, flagged)
  * hierarchical bootstrap over cells (each cell EQUAL weight; resample cells
    then problems within cell) -> pooled Δ + CI, robust to GSM8K size dominance
  * macro / micro / worst-dataset AUROC for ChainUQ and the null
  * macro WITHOUT AMC23 (small-n sensitivity)

Reuses cdyn caches for ChainUQ + CONV features; gen for baselines; sampans for SC.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict, Counter

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from acd.env import EXP_ROOT
from acd import baselines as S

SEEDS = [2026, 7, 13, 42, 100]


def clean(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.isfinite(x).all():
        cmin = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0)
        cmin = np.where(np.isfinite(cmin), cmin, 0.0)
        idx = np.where(~np.isfinite(x))
        x[idx] = np.take(cmin, idx[1])
    return x


def oof(X, y):
    X = clean(X); acc = np.zeros(len(y))
    for s in SEEDS:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            acc[te] += c.predict_proba(X[te])[:, 1]
    return acc / len(SEEDS)


def sc_vote(ans, k):
    a = [x for x in ans[:k] if x is not None]
    if not a:
        return 0.0, 0.0
    c = Counter(a); tot = sum(c.values())
    vote = c.most_common(1)[0][1] / tot
    ent = -sum((v / tot) * np.log(v / tot) for v in c.values())
    return vote, ent


def load(tag):
    cf = EXP_ROOT / "conf" / f"{tag}_conf.json"
    lf = EXP_ROOT / "labels" / f"{tag}.json"
    gf = EXP_ROOT / "gen" / f"{tag}.json"
    cdf = EXP_ROOT / "cdyn" / f"{tag}.json"
    saf = EXP_ROOT / "sampans" / f"{tag}.json"
    ptf = EXP_ROOT / "ptrue" / f"{tag}_ptrue.json"
    if not (cf.exists() and lf.exists() and gf.exists() and cdf.exists()):
        return None
    recs = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
    cdyn = json.loads(cdf.read_text())
    gen = {g["id"]: g for g in json.loads(gf.read_text())}
    sampans = json.loads(saf.read_text()) if saf.exists() else {}
    ptrue = json.loads(ptf.read_text()) if ptf.exists() else None
    CONV, CDYN, SEQ, FINAL, y = [], [], [], [], []
    logp, selfc, pt, votes, ents = [], [], [], [], []
    for r in recs:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in cdyn:
            continue
        CONV.append(cdyn[rid]["conv"]); CDYN.append(cdyn[rid]["cdyn"])
        g = gen.get(rid, {})
        SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        lp = [x.get("neutral_lp") for x in r["intermediate"] if x.get("neutral_lp") is not None]
        FINAL.append([lp[-1] if lp else -10.0])
        logp.append(S.sig_mean_logprob(g)); selfc.append(S.sig_self_certainty(g))
        pt.append(ptrue.get(rid, np.nan) if ptrue else np.nan)
        v, e = sc_vote(sampans.get(rid, []), 8); votes.append(v); ents.append(e)
        y.append(labs[rid])
    y = np.array(y)
    if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0:
        return None
    return dict(CONV=np.array(CONV), CDYN=np.array(CDYN), SEQ=clean(np.array(SEQ)),
                FINAL=np.array(FINAL), y=y, logp=np.array(logp), selfc=np.array(selfc),
                pt=np.array(pt), votes=np.array(votes), ents=np.array(ents))


def boot_p(y, sA, sB, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); sA = np.asarray(sA); sB = np.asarray(sB)
    ip = np.where(y == 1)[0]; ineg = np.where(y == 0)[0]; d = []
    for _ in range(n):
        p = rng.choice(ip, len(ip), True); q = rng.choice(ineg, len(ineg), True)
        ii = np.concatenate([p, q]); yy = y[ii]
        try:
            d.append(roc_auc_score(yy, sA[ii]) - roc_auc_score(yy, sB[ii]))
        except Exception:
            pass
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d <= 0) if (d > 0).mean() >= 0.5 else np.mean(d >= 0))


def holm(pvals):
    """Return Holm-adjusted flags (reject at 0.05)."""
    idx = np.argsort(pvals); m = len(pvals); adj = np.zeros(m); rej = np.zeros(m, bool)
    running = 0.0
    for rank, i in enumerate(idx):
        a = (m - rank) * pvals[i]
        running = max(running, a)
        adj[i] = min(running, 1.0)
    rej = adj < 0.05
    return adj, rej


def hier_boot(cell_data, n=2000, seed=0):
    """Hierarchical bootstrap: resample CELLS (equal weight) then problems.
    cell_data: list of (y, sA, sB). Returns mean Δ of per-cell AUROC diff + CI."""
    rng = np.random.RandomState(seed); K = len(cell_data); ds = []
    for _ in range(n):
        cells = rng.choice(K, K, True); diffs = []
        for ci in cells:
            y, sA, sB = cell_data[ci]
            ip = np.where(y == 1)[0]; ineg = np.where(y == 0)[0]
            if len(ip) == 0 or len(ineg) == 0:
                continue
            p = rng.choice(ip, len(ip), True); q = rng.choice(ineg, len(ineg), True)
            ii = np.concatenate([p, q]); yy = y[ii]
            try:
                diffs.append(roc_auc_score(yy, sA[ii]) - roc_auc_score(yy, sB[ii]))
            except Exception:
                pass
        if diffs:
            ds.append(np.mean(diffs))
    ds = np.array(ds)
    return float(np.mean(ds)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(np.mean(ds <= 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    contrasts = {"C1_chainuq_vs_1x": {}, "C2_ord_vs_CplusF": {}, "C3_fusion_vs_sc8": {}}
    au_store = defaultdict(dict)   # method -> {tag: auroc}
    hier = {k: [] for k in contrasts}

    for tag in args.tags:
        c = load(tag)
        if c is None:
            continue
        y = c["y"]
        chainuq = oof(np.hstack([c["CONV"], c["CDYN"], c["SEQ"]]), y)
        cf = oof(np.hstack([c["CONV"], c["FINAL"]]), y)
        ordtraj = oof(np.hstack([c["CONV"], c["CDYN"]]), y)   # ordered trajectory, no seq
        # strongest comparable 1x baseline per cell
        cand = {"mean_logprob": c["logp"], "self_certainty": c["selfc"]}
        if np.isfinite(c["pt"]).any():
            cand["p_true"] = np.nan_to_num(c["pt"], nan=np.nanmin(c["pt"][np.isfinite(c["pt"])]) if np.isfinite(c["pt"]).any() else 0.0)
        best1x = max(cand, key=lambda k: roc_auc_score(y, clean(cand[k]).ravel()))
        s_best = clean(cand[best1x]).ravel()
        # SC@8 + fusion
        sc8 = c["votes"]
        fusion = oof(np.hstack([c["CONV"], c["CDYN"], c["SEQ"],
                                c["votes"].reshape(-1, 1), c["ents"].reshape(-1, 1)]), y)

        a = lambda s: roc_auc_score(y, s)
        au_store["chainuq"][tag] = a(chainuq); au_store["C+F"][tag] = a(cf)
        au_store["ord_traj"][tag] = a(ordtraj); au_store[f"best1x"][tag] = a(s_best)
        au_store["sc8"][tag] = a(sc8); au_store["fusion"][tag] = a(fusion)
        au_store["best1x_name"][tag] = best1x

        for name, sA, sB in [("C1_chainuq_vs_1x", chainuq, s_best),
                             ("C2_ord_vs_CplusF", ordtraj, cf),
                             ("C3_fusion_vs_sc8", fusion, sc8)]:
            md, lo, hi, p = boot_p(y, sA, sB, n=args.boot)
            contrasts[name][tag] = {"delta": md, "ci": [lo, hi], "p": p,
                                    "auA": a(sA), "auB": a(sB)}
            hier[name].append((y, np.asarray(sA), np.asarray(sB)))

    # Holm per contrast + hierarchical bootstrap
    report = {}
    for name in contrasts:
        tags = list(contrasts[name].keys())
        pv = np.array([contrasts[name][t]["p"] for t in tags])
        adj, rej = holm(pv)
        hm, hlo, hhi, hp = hier_boot(hier[name], n=2000)
        for i, t in enumerate(tags):
            contrasts[name][t]["holm_p"] = float(adj[i]); contrasts[name][t]["holm_sig"] = bool(rej[i])
        deltas = [contrasts[name][t]["delta"] for t in tags]
        report[name] = {"n_cells": len(tags), "mean_delta": float(np.mean(deltas)),
                        "sig_uncorrected": int(sum(contrasts[name][t]["p"] < 0.05 for t in tags)),
                        "sig_holm": int(rej.sum()),
                        "hier_boot": {"mean": hm, "ci": [hlo, hhi], "p": hp},
                        "per_cell": contrasts[name]}

    # macro/micro/worst summaries
    def macro(m, exclude=None):
        vals = [v for t, v in au_store[m].items() if not (exclude and t.startswith(exclude))]
        return float(np.mean(vals))
    def worst(m):
        return float(np.min(list(au_store[m].values())))

    print("\n==== PRIMARY CONTRAST SUMMARY ====")
    for name in contrasts:
        r = report[name]
        hb = r["hier_boot"]
        print(f"\n{name}:")
        print(f"  mean Δ (per-cell) {r['mean_delta']:+.4f} | sig uncorrected {r['sig_uncorrected']}/{r['n_cells']}"
              f" | sig after Holm {r['sig_holm']}/{r['n_cells']}")
        print(f"  hierarchical bootstrap (cells equal weight): Δ {hb['mean']:+.4f} "
              f"CI[{hb['ci'][0]:+.4f},{hb['ci'][1]:+.4f}] p={hb['p']:.4f}")

    print("\n==== AUROC SUMMARIES (macro / macro-no-AMC23 / worst-dataset) ====")
    for m in ["best1x", "C+F", "ord_traj", "chainuq", "sc8", "fusion"]:
        print(f"  {m:12s} macro {macro(m):.3f} | no-amc23 {macro(m, exclude='amc23'):.3f} | worst {worst(m):.3f}")

    if args.out:
        out = {"report": report, "au": {m: au_store[m] for m in au_store if m != 'best1x_name'},
               "best1x_choice": au_store["best1x_name"],
               "summaries": {m: {"macro": macro(m), "macro_no_amc23": macro(m, exclude="amc23"),
                                 "worst": worst(m)} for m in
                             ["best1x", "C+F", "ord_traj", "chainuq", "sc8", "fusion"]}}
        json.dump(out, open(args.out, "w"), indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
