"""PC-CP-Shift: distribution-free selective reasoning whose risk guarantee SURVIVES
domain shift (the KDD contribution).

Finding from conformal_selective.py: naive split-conformal (calibrate tau on pooled
source domains, deploy on an unseen target) has validity ~0.45 at alpha=0.2 for EVERY
method — exchangeability fails across domains, so the guarantee is void off-domain.

Fix: treat DOMAINS as the exchangeable unit (hierarchical / domain-level conformal).
Per held-out target, estimate each source domain's selective risk with a
leave-one-source-domain-out head (honest: the head scoring domain d never trained on
d), then certify the threshold with a bound for a NEW domain's risk:

  naive     : RCPS UCB on pooled source rows (example-level; ignores domain variation)
  worstdom  : tau must satisfy per-domain UCB <= alpha for ALL source domains
  domaincp  : domain-level prediction bound. For candidate tau, per-domain risks r_d
              (d=1..k sources); certify   mean(r_d) + t_{k-1,1-delta} * std(r_d) *
              sqrt(1 + 1/k)  <= alpha  (a 1-delta upper prediction bound for the risk
              of a new, unseen domain). Pick the smallest tau (max coverage) that
              certifies.

Deployed head is trained on ALL source domains and applied to the target. We report
target-domain coverage + validity (fraction of target cells with test risk <= alpha)
for each strategy and method. Claim: domaincp restores validity toward 1-delta while
ChainUQ keeps the highest coverage among 1x same-cost methods.

Output conformal_robust.json.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict, Counter

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from acd.env import EXP_ROOT
from acd import baselines as S

DATASETS = ["gsm8k", "math500", "minerva", "olympiad", "amc23"]

# one-sided t quantiles t_{df, 0.9} (delta=0.1); fallback to 1.5 for larger df
T90 = {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476, 6: 1.440, 7: 1.415}


def t90(df):
    return T90.get(df, 1.34)


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


def parse_tag(tag):
    p = tag.split("_"); return p[0], "_".join(p[1:-1])


def load(tag):
    cf = EXP_ROOT / "conf" / f"{tag}_conf.json"; lf = EXP_ROOT / "labels" / f"{tag}.json"
    gf = EXP_ROOT / "gen" / f"{tag}.json"; cdf = EXP_ROOT / "cdyn" / f"{tag}.json"
    saf = EXP_ROOT / "sampans" / f"{tag}.json"; ptf = EXP_ROOT / "ptrue" / f"{tag}_ptrue.json"
    if not (cf.exists() and lf.exists() and gf.exists() and cdf.exists()):
        return None
    recs = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
    cdyn = json.loads(cdf.read_text()); gen = {g["id"]: g for g in json.loads(gf.read_text())}
    sampans = json.loads(saf.read_text()) if saf.exists() else {}
    ptrue = json.loads(ptf.read_text()) if ptf.exists() else None
    CONV, CDYN, SEQ, y, logp, pt, vote = [], [], [], [], [], [], []
    for r in recs:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in cdyn:
            continue
        CONV.append(cdyn[rid]["conv"]); CDYN.append(cdyn[rid]["cdyn"])
        g = gen.get(rid, {})
        SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        logp.append(S.sig_mean_logprob(g))
        pt.append(ptrue.get(rid, np.nan) if ptrue else np.nan)
        a = [x for x in sampans.get(rid, [])[:8] if x is not None]
        vote.append(Counter(a).most_common(1)[0][1] / len(a) if a else 0.0)
        y.append(labs[rid])
    y = np.array(y)
    if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0:
        return None
    return dict(CONV=np.array(CONV), CDYN=np.array(CDYN), SEQ=clean(np.array(SEQ)),
                y=y, logp=np.array(logp), pt=np.array(pt), vote=np.array(vote))


FEAT = {"chainuq": lambda c: np.hstack([c["CONV"], c["CDYN"], c["SEQ"]]),
        "conv+final": lambda c: np.hstack([c["CONV"], c["CDYN"][:, :1] * 0 + c["CDYN"][:, 1:2]])}
# conv+final approximated by conv + final-confidence (cdyn[1] is last_lp)
FEAT["conv+final"] = lambda c: np.hstack([c["CONV"], c["CDYN"][:, 1:2]])


def fit_score(train_cells, eval_cell, feat):
    Xs = clean(np.vstack([feat(c) for c in train_cells]))
    ys = np.concatenate([c["y"] for c in train_cells])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xs, ys)
    return clf.predict_proba(clean(feat(eval_cell)))[:, 1]


def raw_score(method, cell):
    if method == "logprob":
        return cell["logp"]
    if method == "sc@8":
        return cell["vote"]
    if method == "p_true":
        v = cell["pt"]
        return np.nan_to_num(v, nan=np.nanmin(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0)
    return None


def per_domain_curves(scores_list, y_list, grid):
    """For each domain, selective risk + n_answered at each threshold in grid."""
    curves = []
    for s, y in zip(scores_list, y_list):
        risks, ns = [], []
        for tau in grid:
            ans = s >= tau
            na = int(ans.sum())
            ns.append(na)
            risks.append((1 - y[ans]).mean() if na > 0 else 0.0)
        curves.append((np.array(risks), np.array(ns)))
    return curves


def choose_tau(strategy, curves, grid, alpha, delta=0.1):
    """Return the smallest tau (max coverage) in grid that certifies risk<=alpha."""
    k = len(curves)
    best = np.inf
    for j, tau in enumerate(grid):
        rs = np.array([c[0][j] for c in curves])
        ns = np.array([c[1][j] for c in curves])
        if (ns < 5).any():           # need enough answered per domain to estimate
            continue
        if strategy == "naive":
            tot_err = sum(c[0][j] * c[1][j] for c in curves); tot_n = ns.sum()
            emp = tot_err / max(tot_n, 1)
            ucb = emp + np.sqrt(np.log(1 / delta) / (2 * max(tot_n, 1)))
            ok = ucb <= alpha
        elif strategy == "worstdom":
            ucb = rs + np.sqrt(np.log(1 / delta) / (2 * ns))
            ok = (ucb <= alpha).all()
        elif strategy == "domaincp":
            mu = rs.mean(); sd = rs.std(ddof=1) if k > 1 else 0.0
            bound = mu + t90(k - 1) * sd * np.sqrt(1 + 1.0 / k)
            ok = bound <= alpha
        else:
            raise ValueError(strategy)
        if ok:
            best = min(best, tau)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2])
    ap.add_argument("--methods", nargs="+", default=["logprob", "p_true", "sc@8", "chainuq"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cells = {}
    for tag in args.tags:
        c = load(tag)
        if c is None:
            continue
        ds, m = parse_tag(tag); cells[(ds, m)] = c
    by_model = defaultdict(dict)
    for (ds, m), c in cells.items():
        by_model[m][ds] = c

    grid = np.linspace(0.0, 1.0, 101)
    STRAT = ["naive", "worstdom", "domaincp"]
    agg = {a: {st: {m: {"cov": [], "valid": []} for m in args.methods} for st in STRAT}
           for a in args.alphas}
    detail = []

    for model, dss in sorted(by_model.items()):
        have = [d for d in DATASETS if d in dss]
        if len(have) < 3:
            continue
        for tgt in have:
            srcs = [d for d in have if d != tgt]
            tgt_c = dss[tgt]
            for m in args.methods:
                # source per-domain scores via leave-one-source-domain-out (honest)
                sdom_scores, sdom_y = [], []
                if m in FEAT:
                    for d in srcs:
                        tr = [dss[o] for o in srcs if o != d]
                        if not tr:
                            sdom_scores.append(raw_score("logprob", dss[d])); sdom_y.append(dss[d]["y"]); continue
                        sdom_scores.append(fit_score(tr, dss[d], FEAT[m])); sdom_y.append(dss[d]["y"])
                    s_tgt = fit_score([dss[d] for d in srcs], tgt_c, FEAT[m])
                else:
                    for d in srcs:
                        sdom_scores.append(raw_score(m, dss[d])); sdom_y.append(dss[d]["y"])
                    s_tgt = raw_score(m, tgt_c)
                curves = per_domain_curves(sdom_scores, sdom_y, grid)
                for a in args.alphas:
                    for st in STRAT:
                        tau = choose_tau(st, curves, grid, a)
                        ans = s_tgt >= tau
                        cov = float(ans.mean())
                        risk = float((1 - tgt_c["y"][ans]).mean()) if ans.sum() > 0 else np.nan
                        agg[a][st][m]["cov"].append(cov)
                        if not np.isnan(risk):
                            agg[a][st][m]["valid"].append(risk <= a)
                        if m == "chainuq" and a == 0.2:
                            detail.append({"model": model, "target": tgt, "strategy": st,
                                           "coverage": cov, "risk": risk})

    print("\n=== PC-CP UNDER SHIFT: naive vs domain-robust calibration ===")
    print("validity = fraction of unseen target domains with test risk <= alpha (want >= 0.9)")
    for a in args.alphas:
        print(f"\n alpha={a}")
        print(f"   {'strategy':10s}{'method':12s}{'coverage':>10s}{'validity':>10s}")
        for st in STRAT:
            for m in args.methods:
                d = agg[a][st][m]
                cov = np.mean(d["cov"]) if d["cov"] else 0.0
                val = np.mean(d["valid"]) if d["valid"] else float("nan")
                print(f"   {st:10s}{m:12s}{cov:10.3f}{val:10.2f}")
            print()
    if args.out:
        out = {"summary": {str(a): {st: {m: {
            "coverage": float(np.mean(agg[a][st][m]["cov"])) if agg[a][st][m]["cov"] else 0.0,
            "validity": float(np.mean(agg[a][st][m]["valid"])) if agg[a][st][m]["valid"] else None}
            for m in args.methods} for st in STRAT} for a in args.alphas},
            "chainuq_detail_a20": detail}
        json.dump(out, open(args.out, "w"), indent=2); print("saved", args.out)


if __name__ == "__main__":
    main()
