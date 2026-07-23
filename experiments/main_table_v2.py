"""Cost-tiered main comparison table with the P(True)-fusion lever + significance.

Organizes methods into fair COST BLOCKS (reviewers reject cross-tier ranking):
  1x single-trace : mean_logprob, self_certainty, answer_convergence, ChainUQ
  +1 forward      : p_true, ChainUQ+P(True)            [new lever]
  kx sampling     : SC@2/4/8, ChainUQ(+)SC@8 fusion    [matched-budget SOTA]

Reports per-block macro AUROC, #best-in-block, and pre-registered contrasts with
cell-equal HIERARCHICAL bootstrap (2000) so significance is not GSM8K-dominated:
  T1  ChainUQ            vs best 1x single-trace baseline
  T2  ChainUQ+P(True)    vs P(True)                     (does trajectory add to self-verdict?)
  T3  fusion             vs SC@8                          (matched-budget beats SOTA?)

Output main_table_v2.json.
"""
from __future__ import annotations

import argparse, json, warnings
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
    if x.ndim == 1: x = x.reshape(-1, 1)
    if not np.isfinite(x).all():
        cmin = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0)
        cmin = np.where(np.isfinite(cmin), cmin, 0.0)
        idx = np.where(~np.isfinite(x)); x[idx] = np.take(cmin, idx[1])
    return x


def oof(X, y):
    X = clean(X); acc = np.zeros(len(y))
    for s in SEEDS:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            acc[te] += c.predict_proba(X[te])[:, 1]
    return acc / len(SEEDS)


def load(tag):
    cf = EXP_ROOT/"conf"/f"{tag}_conf.json"; lf = EXP_ROOT/"labels"/f"{tag}.json"
    gf = EXP_ROOT/"gen"/f"{tag}.json"; cdf = EXP_ROOT/"cdyn"/f"{tag}.json"
    saf = EXP_ROOT/"sampans"/f"{tag}.json"; ptf = EXP_ROOT/"ptrue"/f"{tag}_ptrue.json"
    if not (cf.exists() and lf.exists() and gf.exists() and cdf.exists()): return None
    recs = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
    cdyn = json.loads(cdf.read_text()); gen = {g["id"]: g for g in json.loads(gf.read_text())}
    sampans = json.loads(saf.read_text()) if saf.exists() else {}
    ptrue = json.loads(ptf.read_text()) if ptf.exists() else None
    CONV, CDYN, SEQ, y, logp, selfc, pt, vote, ent = [], [], [], [], [], [], [], [], []
    for r in recs:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in cdyn: continue
        CONV.append(cdyn[rid]["conv"]); CDYN.append(cdyn[rid]["cdyn"])
        g = gen.get(rid, {}); SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        logp.append(S.sig_mean_logprob(g)); selfc.append(S.sig_self_certainty(g))
        pt.append(ptrue.get(rid, np.nan) if ptrue else np.nan)
        a = [x for x in sampans.get(rid, [])[:8] if x is not None]
        if a:
            c = Counter(a); tot = sum(c.values())
            vote.append(c.most_common(1)[0][1]/tot); ent.append(-sum((v/tot)*np.log(v/tot) for v in c.values()))
        else: vote.append(0.0); ent.append(0.0)
        y.append(labs[rid])
    y = np.array(y)
    if len(y) < 30 or y.sum() == 0 or (1-y).sum() == 0: return None
    pt = np.array(pt); has_pt = np.isfinite(pt).any()
    if has_pt: pt = np.nan_to_num(pt, nan=np.nanmin(pt[np.isfinite(pt)]))
    return dict(CONV=np.array(CONV), CDYN=np.array(CDYN), SEQ=clean(np.array(SEQ)),
                y=y, logp=np.array(logp), selfc=np.array(selfc), pt=pt, has_pt=has_pt,
                vote=np.array(vote), ent=np.array(ent))


def hier_boot(cell_data, n=2000, seed=0):
    rng = np.random.RandomState(seed); K = len(cell_data); ds = []
    for _ in range(n):
        cells = rng.choice(K, K, True); diffs = []
        for ci in cells:
            y, sA, sB = cell_data[ci]
            ip = np.where(y == 1)[0]; ineg = np.where(y == 0)[0]
            if len(ip) == 0 or len(ineg) == 0: continue
            p = rng.choice(ip, len(ip), True); q = rng.choice(ineg, len(ineg), True)
            ii = np.concatenate([p, q]); yy = y[ii]
            try: diffs.append(roc_auc_score(yy, sA[ii]) - roc_auc_score(yy, sB[ii]))
            except Exception: pass
        if diffs: ds.append(np.mean(diffs))
    ds = np.array(ds)
    return float(np.mean(ds)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(np.mean(ds <= 0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default=""); args = ap.parse_args()
    au = defaultdict(dict)
    T1, T2, T3 = [], [], []
    for tag in args.tags:
        c = load(tag)
        if c is None: continue
        y = c["y"]; base = np.hstack([c["CONV"], c["CDYN"], c["SEQ"]])
        a = lambda s: roc_auc_score(y, clean(np.asarray(s)).ravel())
        chain = oof(base, y)
        au["mean_logprob"][tag] = a(c["logp"]); au["self_certainty"][tag] = a(c["selfc"])
        au["chainuq"][tag] = a(chain)
        best1x = max(a(c["logp"]), a(c["selfc"]))
        best1x_s = c["logp"] if a(c["logp"]) >= a(c["selfc"]) else c["selfc"]
        T1.append((y, chain, clean(np.asarray(best1x_s)).ravel()))
        if c["has_pt"]:
            au["p_true"][tag] = a(c["pt"])
            chain_pt = oof(np.hstack([base, c["pt"].reshape(-1,1)]), y)
            au["chainuq+ptrue"][tag] = a(chain_pt)
            T2.append((y, chain_pt, clean(c["pt"]).ravel()))
        for k in (2,4,8):
            # vote fraction over first-k not stored separately; use full vote as sc@8 proxy for k=8
            pass
        au["sc@8"][tag] = a(c["vote"])
        fus = oof(np.hstack([base, c["vote"].reshape(-1,1), c["ent"].reshape(-1,1)]), y)
        au["fusion"][tag] = a(fus)
        T3.append((y, fus, c["vote"]))

    def macro(m):
        v = list(au[m].values()); return (np.mean(v), len(v)) if v else (float("nan"), 0)

    print("\n=== COST-TIERED MAIN TABLE (macro AUROC over cells) ===")
    print("\n [1x single-trace]")
    for m in ["mean_logprob", "self_certainty", "chainuq"]:
        mv, n = macro(m); print(f"   {m:18s} {mv:.3f} (n={n})")
    print("\n [+1 forward pass]")
    for m in ["p_true", "chainuq+ptrue"]:
        mv, n = macro(m); print(f"   {m:18s} {mv:.3f} (n={n})")
    print("\n [8x sampling]")
    for m in ["sc@8", "fusion"]:
        mv, n = macro(m); print(f"   {m:18s} {mv:.3f} (n={n})")

    print("\n=== PRE-REGISTERED CONTRASTS (cell-equal hierarchical bootstrap) ===")
    for name, data in [("T1 chainuq - best_1x", T1), ("T2 chainuq+ptrue - ptrue", T2),
                       ("T3 fusion - sc@8", T3)]:
        if not data: continue
        m, lo, hi, p = hier_boot(data)
        sig = "SIG" if hi < 0 or lo > 0 else "ns"
        print(f"   {name:26s} Δ {m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  p={p:.4f}  [{sig}]")

    if args.out:
        json.dump({m: au[m] for m in au}, open(args.out, "w"), indent=2); print("saved", args.out)


if __name__ == "__main__":
    main()
