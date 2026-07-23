"""Adaptive selective-prediction frontier: does ChainUQ-guided budget allocation
beat uniform self-consistency on the AUROC-vs-cost frontier?

This is NOT accuracy routing (that was a dead end; P(True) routes accuracy better).
It is a UQ frontier: at a fixed average sampling budget, produce the best correctness
RANKING. Every question first pays ~1x for the primary trace + ChainUQ probe. Then a
fraction rho of questions -- chosen by LOWEST ChainUQ confidence -- are escalated to k
full samples; the rest keep their 1x ChainUQ score. Escalated questions use a fusion
score (ChainUQ features + SC vote/entropy). All scores are P(correct) in [0,1] (OOF
logistic), so they rank on a common scale.

  avg cost(rho) = (1-rho)*1 + rho*k        (unsampled cost 1, sampled cost k)

Compare adaptive AUROC(rho) against:
  uniform   : SC@m at the SAME avg cost m (interpolated over m in {1,2,4,8})
  random    : escalate a RANDOM rho fraction (20 seeds) -- isolates the value of the
              ChainUQ routing decision from the value of extra samples alone.

Positive result = adaptive frontier lies above both uniform and random at matched cost.
Output adaptive_frontier.json.
"""
from __future__ import annotations

import argparse, json, warnings
from collections import Counter, defaultdict
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from acd.env import EXP_ROOT
from acd import baselines as S

SEEDS = [2026, 7, 13]


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


def vote_frac(ans, k):
    a = [x for x in ans[:k] if x is not None]
    if not a: return 0.0, 0.0
    c = Counter(a); tot = sum(c.values())
    return c.most_common(1)[0][1] / tot, -sum((v/tot)*np.log(v/tot) for v in c.values())


def load(tag):
    cf = EXP_ROOT/"conf"/f"{tag}_conf.json"; lf = EXP_ROOT/"labels"/f"{tag}.json"
    gf = EXP_ROOT/"gen"/f"{tag}.json"; cdf = EXP_ROOT/"cdyn"/f"{tag}.json"
    saf = EXP_ROOT/"sampans"/f"{tag}.json"
    if not (cf.exists() and lf.exists() and gf.exists() and cdf.exists() and saf.exists()):
        return None
    recs = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
    cdyn = json.loads(cdf.read_text()); gen = {g["id"]: g for g in json.loads(gf.read_text())}
    sampans = json.loads(saf.read_text())
    CONV, CDYN, SEQ, y, ans = [], [], [], [], []
    for r in recs:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in cdyn: continue
        CONV.append(cdyn[rid]["conv"]); CDYN.append(cdyn[rid]["cdyn"])
        g = gen.get(rid, {}); SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        ans.append(sampans.get(rid, [])); y.append(labs[rid])
    y = np.array(y)
    if len(y) < 40 or y.sum() == 0 or (1-y).sum() == 0: return None
    return dict(CONV=np.array(CONV), CDYN=np.array(CDYN), SEQ=clean(np.array(SEQ)), y=y, ans=ans)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--out", default="")
    args = ap.parse_args()
    K = args.k
    RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    # aggregate AUROC by cost bin
    adaptive = defaultdict(list); randomr = defaultdict(list)
    uniform = defaultdict(list)  # cost m -> auroc
    for tag in args.tags:
        c = load(tag)
        if c is None: continue
        y = c["y"]; n = len(y)
        base = np.hstack([c["CONV"], c["CDYN"], c["SEQ"]])
        p_chain = oof(base, y)                                   # 1x P(correct)
        votes = np.array([vote_frac(a, K)[0] for a in c["ans"]])
        ents = np.array([vote_frac(a, K)[1] for a in c["ans"]])
        p_fus = oof(np.hstack([base, votes.reshape(-1,1), ents.reshape(-1,1)]), y)  # kx fused
        # uniform SC@m reference points
        for m in [1, 2, 4, 8]:
            if m == 1:
                uniform[1.0].append(roc_auc_score(y, p_chain))   # 1x = ChainUQ
            else:
                vm = np.array([vote_frac(a, m)[0] for a in c["ans"]])
                uniform[float(m)].append(roc_auc_score(y, vm))
        # adaptive: escalate lowest-confidence rho fraction
        order = np.argsort(p_chain)  # ascending: least confident first
        for rho in RHOS:
            nesc = int(round(rho * n))
            esc = set(order[:nesc].tolist())
            score = np.array([p_fus[i] if i in esc else p_chain[i] for i in range(n)])
            cost = (1 - rho) * 1 + rho * K
            adaptive[round(cost, 3)].append((roc_auc_score(y, score), rho))
            # random routing control
            raucs = []
            for sd in range(20):
                rng = np.random.RandomState(sd)
                resc = set(rng.permutation(n)[:nesc].tolist())
                rsc = np.array([p_fus[i] if i in resc else p_chain[i] for i in range(n)])
                raucs.append(roc_auc_score(y, rsc))
            randomr[round(cost, 3)].append(np.mean(raucs))

    # report frontier
    print(f"\n=== ADAPTIVE SELECTIVE-PREDICTION FRONTIER (k={K}, {len(adaptive[1.0]) if 1.0 in adaptive else '?'} cells) ===")
    print("uniform SC reference (avg cost -> macro AUROC):")
    for m in sorted(uniform):
        print(f"   cost {m:4.1f}x : {np.mean(uniform[m]):.3f}")
    print("\nadaptive vs random routing at matched avg cost:")
    print(f"   {'cost':>6s}{'rho':>6s}{'adaptive':>10s}{'random':>9s}{'Δ(adapt-rand)':>14s}{'uniform@cost':>13s}{'Δ(adapt-unif)':>14s}")
    # interpolate uniform curve for arbitrary cost
    umx = sorted(uniform); uy = [np.mean(uniform[m]) for m in umx]
    def unif_at(cost): return float(np.interp(cost, umx, uy))
    rows = []
    for cost in sorted(adaptive):
        a_auc = np.mean([v[0] for v in adaptive[cost]]); rho = adaptive[cost][0][1]
        r_auc = np.mean(randomr[cost]); u_auc = unif_at(cost)
        print(f"   {cost:6.2f}{rho:6.2f}{a_auc:10.3f}{r_auc:9.3f}{a_auc-r_auc:+14.3f}{u_auc:13.3f}{a_auc-u_auc:+14.3f}")
        rows.append({"cost": cost, "rho": rho, "adaptive": a_auc, "random": r_auc,
                     "uniform_interp": u_auc, "d_random": a_auc-r_auc, "d_uniform": a_auc-u_auc})
    # headline: best cost-saving to reach SC@8 quality
    sc8 = np.mean(uniform[8.0])
    reach = next((r for r in rows if r["adaptive"] >= sc8 - 1e-9), None)
    if reach:
        print(f"\n   adaptive reaches SC@8 quality ({sc8:.3f}) at avg cost {reach['cost']:.2f}x (vs 8x), rho={reach['rho']:.2f}")
    else:
        print(f"\n   adaptive does NOT reach SC@8 quality ({sc8:.3f}) below 8x; best {max(r['adaptive'] for r in rows):.3f}")
    if args.out:
        json.dump({"uniform": {str(m): float(np.mean(uniform[m])) for m in uniform}, "frontier": rows},
                  open(args.out, "w"), indent=2); print("saved", args.out)


if __name__ == "__main__":
    main()
