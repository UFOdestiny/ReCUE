"""Measure which levers actually improve ChainUQ performance (honest 5-seed OOF).

Two cost tiers, macro AUROC over 31 math cells.

1x TIER (goal: push single-generation 0.791 up):
  base            CONV+CDYN+SEQ, logistic          (current ChainUQ)
  +histgbt        same features, HistGradientBoosting
  +rf             same features, RandomForest
  +ptrue_feat     base features + P(True) as an extra feature (still 1 fwd, so 1x+1fwd)
  +richfeat       base + extra trajectory stats (quantiles, autocorr, rises, argpos)
  +ptrue+rich+gbt combine the winning levers

SAMPLING TIER (goal: match/beat SC@8 at 8x):
  sc@8            vote fraction                     (the reference SOTA)
  fusion          ChainUQ features + SC vote + SC entropy, logistic (matched 8x)
  fusion+gbt      same, HistGBT
  fusion+rich+gbt richfeat + SC stats, HistGBT

Report macro, #cells-best, and Δ vs the relevant null. Levers that don't beat the
null on a paired sense are reported honestly (no cherry-picking).
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore")
_trapz = getattr(np, "trapezoid", np.trapz)

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from acd.env import EXP_ROOT
from acd import baselines as S
from acd.features import _eq

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


def head(kind):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=5,
                                      random_state=0, n_jobs=4)
    if kind == "gbt":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08,
                                              max_iter=300, l2_regularization=1.0,
                                              random_state=0)
    raise ValueError(kind)


def oof(X, y, kind="logistic", seeds=SEEDS):
    X = clean(X); acc = np.zeros(len(y))
    for s in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = head(kind).fit(X[tr], y[tr])
            acc[te] += c.predict_proba(X[te])[:, 1]
    return acc / len(seeds)


def rich_traj(rec):
    """extra order-invariant + shape stats from the confidence trajectory."""
    lp = [x.get("neutral_lp") for x in rec["intermediate"] if x.get("neutral_lp") is not None]
    if not lp:
        return [-10.0] * 8
    a = np.asarray(lp, float); n = len(a)
    q25, q75 = np.percentile(a, 25), np.percentile(a, 75)
    nrise = sum(1 for i in range(1, n) if a[i] > a[i - 1]) / max(1, n - 1)
    area = float(_trapz(a) / max(n - 1, 1))
    ac = float(np.corrcoef(a[:-1], a[1:])[0, 1]) if n >= 2 and a.std() > 1e-9 else 0.0
    if not np.isfinite(ac):
        ac = 0.0
    argmin = float(np.argmin(a) / max(n - 1, 1))
    return [float(q25), float(q75), float(a.std() if n > 1 else 0), nrise, area, ac, argmin,
            float(a.max() - a.min())]


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
    CONV, CDYN, SEQ, RICH, y, pt, vote, ent = [], [], [], [], [], [], [], []
    for r in recs:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in cdyn:
            continue
        CONV.append(cdyn[rid]["conv"]); CDYN.append(cdyn[rid]["cdyn"])
        g = gen.get(rid, {})
        SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        RICH.append(rich_traj(r))
        pt.append(ptrue.get(rid, np.nan) if ptrue else np.nan)
        a = [x for x in sampans.get(rid, [])[:8] if x is not None]
        if a:
            c = Counter(a); tot = sum(c.values())
            vote.append(c.most_common(1)[0][1] / tot)
            ent.append(-sum((v / tot) * np.log(v / tot) for v in c.values()))
        else:
            vote.append(0.0); ent.append(0.0)
        y.append(labs[rid])
    y = np.array(y)
    if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0:
        return None
    pt = np.array(pt)
    has_pt = np.isfinite(pt).any()
    if has_pt:
        pt = np.nan_to_num(pt, nan=np.nanmin(pt[np.isfinite(pt)]))
    return dict(CONV=np.array(CONV), CDYN=np.array(CDYN), SEQ=clean(np.array(SEQ)),
                RICH=clean(np.array(RICH)), y=y, pt=pt, has_pt=has_pt,
                vote=np.array(vote).reshape(-1, 1), ent=np.array(ent).reshape(-1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    au = defaultdict(dict)   # method -> {tag: auroc}
    for tag in args.tags:
        c = load(tag)
        if c is None:
            continue
        y = c["y"]
        base = np.hstack([c["CONV"], c["CDYN"], c["SEQ"]])
        rich = np.hstack([base, c["RICH"]])
        a = lambda s: roc_auc_score(y, s)

        au["base(logistic)"][tag] = a(oof(base, y, "logistic"))
        au["base+gbt"][tag] = a(oof(base, y, "gbt"))
        au["base+rf"][tag] = a(oof(base, y, "rf"))
        au["base+rich(logistic)"][tag] = a(oof(rich, y, "logistic"))
        au["base+rich+gbt"][tag] = a(oof(rich, y, "gbt"))
        if c["has_pt"]:
            bp = np.hstack([base, c["pt"].reshape(-1, 1)])
            au["base+ptrue"][tag] = a(oof(bp, y, "logistic"))
            brp = np.hstack([rich, c["pt"].reshape(-1, 1)])
            au["base+rich+ptrue+gbt"][tag] = a(oof(brp, y, "gbt"))
        # sampling tier
        au["sc@8"][tag] = a(c["vote"].ravel())
        fus = np.hstack([base, c["vote"], c["ent"]])
        au["fusion(logistic)"][tag] = a(oof(fus, y, "logistic"))
        au["fusion+gbt"][tag] = a(oof(fus, y, "gbt"))
        fusr = np.hstack([rich, c["vote"], c["ent"]])
        au["fusion+rich+gbt"][tag] = a(oof(fusr, y, "gbt"))

    def macro(m):
        v = list(au[m].values()); return np.mean(v) if v else float("nan"), len(v)

    print("\n=== 1x TIER (goal: beat base 0.791) ===")
    order1 = ["base(logistic)", "base+gbt", "base+rf", "base+rich(logistic)",
              "base+rich+gbt", "base+ptrue", "base+rich+ptrue+gbt"]
    b, _ = macro("base(logistic)")
    for m in order1:
        mv, n = macro(m)
        # paired win-rate vs base on shared cells
        shared = [t for t in au[m] if t in au["base(logistic)"]]
        wins = sum(1 for t in shared if au[m][t] > au["base(logistic)"][t])
        print(f"  {m:24s} macro {mv:.3f}  Δbase {mv-b:+.3f}  wins {wins}/{len(shared)}  (n={n})")

    print("\n=== SAMPLING TIER (goal: beat sc@8) ===")
    s8, _ = macro("sc@8")
    for m in ["sc@8", "fusion(logistic)", "fusion+gbt", "fusion+rich+gbt"]:
        mv, n = macro(m)
        shared = [t for t in au[m] if t in au["sc@8"]]
        wins = sum(1 for t in shared if au[m][t] > au["sc@8"][t])
        print(f"  {m:24s} macro {mv:.3f}  Δsc@8 {mv-s8:+.3f}  wins {wins}/{len(shared)}  (n={n})")

    if args.out:
        json.dump({m: au[m] for m in au}, open(args.out, "w"), indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
