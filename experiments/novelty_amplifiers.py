"""P1 novelty amplifiers + capacity control (More_EXP.md Sec 4 + Block P0-4 capacity).

Three sub-experiments over the cached 31-cell matrix (all judge-free):

--exp stress   P1-1 self-consistency blind-spot STRESS TEST (pre-registered).
   Thresholds fixed a priori: SC vote-fraction >= {0.625, 0.75, 0.875, 1.0}.
   Also stratify by SC-entropy quantile (low/med/high). On each high-consensus
   subset report n, acc, #wrong, SC-AUROC (~0.5 by construction), ReCUE AUROC,
   and the matched-8x fusion gain. Statement: in near-constant-SC regions,
   does ReCUE still rank confident-consensus errors?

--exp labeleff P1-3 label efficiency. Train ReCUE / seq-only / CONV+FINAL on
   {1,2,5,10,25,50,100}% of a cell's labeled problems (fixed test fold), report
   AUROC vs #labeled examples + labels needed to reach 95% of full-data AUROC.
   Also a pooled cross-cell head to test if multi-domain data cuts target labels.

--exp capacity P0-4 classifier-capacity control. Same ReCUE features, swap the
   head: logistic / RandomForest / GradientBoosting / MLP. Shows the gain is the
   OBSERVATION (features), not classifier capacity.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from recue.env import EXP_ROOT
from recue import baselines as S

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


def make_head(kind):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, n_jobs=-1)
    if kind == "gbt":
        return GradientBoostingClassifier(random_state=0)
    if kind == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0))
    raise ValueError(kind)


def oof(X, y, kind="logistic", seeds=SEEDS):
    X = clean(X); acc = np.zeros(len(y))
    for s in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_head(kind).fit(X[tr], y[tr])
            acc[te] += c.predict_proba(X[te])[:, 1]
    return acc / len(seeds)


def sc_stats(ans, k=8):
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
    if not (cf.exists() and lf.exists() and gf.exists() and cdf.exists()):
        return None
    recs = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
    cdyn = json.loads(cdf.read_text())
    gen = {g["id"]: g for g in json.loads(gf.read_text())}
    sampans = json.loads(saf.read_text()) if saf.exists() else {}
    CONV, CDYN, SEQ, FINAL, y, votes, ents = [], [], [], [], [], [], []
    for r in recs:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in cdyn:
            continue
        CONV.append(cdyn[rid]["conv"]); CDYN.append(cdyn[rid]["cdyn"])
        g = gen.get(rid, {})
        SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        lp = [x.get("neutral_lp") for x in r["intermediate"] if x.get("neutral_lp") is not None]
        FINAL.append([lp[-1] if lp else -10.0])
        v, e = sc_stats(sampans.get(rid, [])); votes.append(v); ents.append(e)
        y.append(labs[rid])
    y = np.array(y)
    if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0:
        return None
    return dict(CONV=np.array(CONV), CDYN=np.array(CDYN), SEQ=clean(np.array(SEQ)),
                FINAL=np.array(FINAL), y=y, votes=np.array(votes), ents=np.array(ents))


def recue_feats(c):
    return np.hstack([c["CONV"], c["CDYN"], c["SEQ"]])


# --------------------------- P1-1 stress test ---------------------------

def exp_stress(tags, out):
    THRESH = [0.625, 0.75, 0.875, 1.0]
    rows = []
    agg = defaultdict(list)
    for tag in tags:
        c = load(tag)
        if c is None:
            continue
        y = c["y"]; votes = c["votes"]
        cq = oof(recue_feats(c), y)
        fusion = oof(np.hstack([recue_feats(c), c["votes"].reshape(-1, 1),
                                c["ents"].reshape(-1, 1)]), y)
        for th in THRESH:
            mask = votes >= th
            n = int(mask.sum())
            if n < 30:
                continue
            ys = y[mask]
            nwrong = int((1 - ys).sum())
            if ys.sum() == 0 or nwrong == 0:
                continue
            try:
                sc_au = roc_auc_score(ys, votes[mask])
            except Exception:
                sc_au = float("nan")
            cq_au = roc_auc_score(ys, cq[mask])
            fu_au = roc_auc_score(ys, fusion[mask])
            rows.append({"tag": tag, "thresh": th, "n": n, "acc": float(ys.mean()),
                         "n_wrong": nwrong, "sc_auroc": sc_au, "recue_auroc": cq_au,
                         "fusion_auroc": fu_au})
            agg[th].append((n, cq_au, fu_au, nwrong))
    print("\n=== P1-1 SELF-CONSISTENCY BLIND-SPOT STRESS TEST ===")
    print("(SC AUROC ~0.5 by construction on high-consensus subsets)")
    print(f"{'vote>=':>8s}{'cells':>7s}{'tot_n':>8s}{'tot_wrong':>10s}{'ReCUE':>9s}{'fusion':>8s}")
    for th in THRESH:
        v = agg[th]
        if not v:
            continue
        totn = sum(x[0] for x in v); totw = sum(x[3] for x in v)
        mcq = np.mean([x[1] for x in v]); mfu = np.mean([x[2] for x in v])
        print(f"{th:8.3f}{len(v):7d}{totn:8d}{totw:10d}{mcq:9.3f}{mfu:8.3f}")
    if out:
        json.dump({"rows": rows}, open(out, "w"), indent=2); print("saved", out)


# --------------------------- P1-3 label efficiency ---------------------------

def exp_labeleff(tags, out):
    FRACS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
    FEATS = {"recue": recue_feats,
             "seq-only": lambda c: c["SEQ"],
             "conv+final": lambda c: np.hstack([c["CONV"], c["FINAL"]])}
    curves = {f: defaultdict(list) for f in FEATS}
    per_cell = []
    for tag in tags:
        c = load(tag)
        if c is None:
            continue
        y = c["y"]; n = len(y)
        # fixed 40% test fold, train subset drawn from remaining 60%
        rng = np.random.RandomState(0)
        idx = rng.permutation(n); nte = int(0.4 * n)
        te = idx[:nte]; pool = idx[nte:]
        yte = y[te]
        if yte.sum() == 0 or (1 - yte).sum() == 0:
            continue
        cell = {"tag": tag, "n_pool": len(pool)}
        for fs, fn in FEATS.items():
            X = clean(fn(c))
            aucs = {}
            for fr in FRACS:
                ntr = max(20, int(fr * len(pool)))
                ntr = min(ntr, len(pool))
                accs = []
                for sd in range(5):
                    r2 = np.random.RandomState(100 + sd)
                    sub = pool[r2.permutation(len(pool))[:ntr]]
                    ysub = y[sub]
                    if ysub.sum() == 0 or (1 - ysub).sum() == 0:
                        continue
                    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[sub], ysub)
                    accs.append(roc_auc_score(yte, clf.predict_proba(X[te])[:, 1]))
                if accs:
                    aucs[fr] = float(np.mean(accs))
                    curves[fs][fr].append(np.mean(accs))
            cell[fs] = aucs
        per_cell.append(cell)
    print("\n=== P1-3 LABEL EFFICIENCY (macro AUROC vs label fraction) ===")
    print(f"{'frac':>7s}" + "".join(f"{f:>12s}" for f in FEATS))
    for fr in FRACS:
        print(f"{fr:7.2f}" + "".join(f"{np.mean(curves[f][fr]):12.3f}" if curves[f][fr] else f"{'-':>12s}" for f in FEATS))
    # labels to reach 95% of full-data AUROC for recue
    full = np.mean(curves["recue"][1.0])
    target = 0.95 * full
    reach = next((fr for fr in FRACS if curves["recue"][fr] and np.mean(curves["recue"][fr]) >= target), 1.0)
    print(f"ReCUE full-data macro {full:.3f}; reaches 95% ({target:.3f}) at ~{reach*100:.0f}% labels")
    if out:
        json.dump({"macro": {f: {str(fr): float(np.mean(curves[f][fr])) if curves[f][fr] else None
                                 for fr in FRACS} for f in FEATS},
                   "per_cell": per_cell, "reach95_frac": reach}, open(out, "w"), indent=2)
        print("saved", out)


# --------------------------- P0-4 capacity control ---------------------------

def exp_capacity(tags, out):
    HEADS = ["logistic", "rf", "gbt", "mlp"]
    agg = {h: [] for h in HEADS}
    rows = []
    for tag in tags:
        c = load(tag)
        if c is None:
            continue
        y = c["y"]; X = recue_feats(c)
        row = {"tag": tag}
        for h in HEADS:
            au = roc_auc_score(y, oof(X, y, kind=h, seeds=[2026, 7, 13]))
            row[h] = au; agg[h].append(au)
        rows.append(row)
    print("\n=== P0-4 CLASSIFIER-CAPACITY CONTROL (same ReCUE features) ===")
    print(f"{'head':>10s}{'macro AUROC':>14s}")
    for h in HEADS:
        print(f"{h:>10s}{np.mean(agg[h]):14.3f}")
    print("Interpretation: gain comes from the OBSERVATION not classifier capacity"
          " if heads are close.")
    if out:
        json.dump({"macro": {h: float(np.mean(agg[h])) for h in HEADS}, "rows": rows},
                  open(out, "w"), indent=2); print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--exp", choices=["stress", "labeleff", "capacity", "all"], default="all")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()
    op = lambda n: (f"{args.out_prefix}_{n}.json" if args.out_prefix else "")
    if args.exp in ("stress", "all"):
        exp_stress(args.tags, op("stress"))
    if args.exp in ("labeleff", "all"):
        exp_labeleff(args.tags, op("labeleff"))
    if args.exp in ("capacity", "all"):
        exp_capacity(args.tags, op("capacity"))


if __name__ == "__main__":
    main()
