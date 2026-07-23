"""Does RandomForest's in-domain +0.035 SURVIVE domain shift (LODO)?

If RF's gain is real structure it should also transfer; if it's in-domain capacity
overfit it will shrink or reverse under LODO. Per backbone, train head on 4 source
math datasets, test on the held-out 5th. Compare logistic vs RF on the SAME ChainUQ
features. Also report +P(True) (concatenated as a feature, logistic) transfer.

Output transfer_head_capacity.json.
"""
from __future__ import annotations

import argparse, json, warnings
from collections import defaultdict
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from acd.env import EXP_ROOT
from acd import baselines as S

DATASETS = ["gsm8k", "math500", "minerva", "olympiad", "amc23"]


def clean(x):
    x = np.asarray(x, float)
    if x.ndim == 1: x = x.reshape(-1, 1)
    if not np.isfinite(x).all():
        cmin = np.nanmin(np.where(np.isfinite(x), x, np.nan), axis=0)
        cmin = np.where(np.isfinite(cmin), cmin, 0.0)
        idx = np.where(~np.isfinite(x)); x[idx] = np.take(cmin, idx[1])
    return x


def parse_tag(t): p = t.split("_"); return p[0], "_".join(p[1:-1])


def load(tag):
    cf = EXP_ROOT/"conf"/f"{tag}_conf.json"; lf = EXP_ROOT/"labels"/f"{tag}.json"
    gf = EXP_ROOT/"gen"/f"{tag}.json"; cdf = EXP_ROOT/"cdyn"/f"{tag}.json"
    ptf = EXP_ROOT/"ptrue"/f"{tag}_ptrue.json"
    if not (cf.exists() and lf.exists() and gf.exists() and cdf.exists()): return None
    recs = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
    cdyn = json.loads(cdf.read_text()); gen = {g["id"]: g for g in json.loads(gf.read_text())}
    ptrue = json.loads(ptf.read_text()) if ptf.exists() else None
    CONV, CDYN, SEQ, PT, y = [], [], [], [], []
    for r in recs:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in cdyn: continue
        CONV.append(cdyn[rid]["conv"]); CDYN.append(cdyn[rid]["cdyn"])
        g = gen.get(rid, {}); SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        PT.append(ptrue.get(rid, np.nan) if ptrue else np.nan)
        y.append(labs[rid])
    y = np.array(y)
    if len(y) < 20 or y.sum() == 0 or (1-y).sum() == 0: return None
    PT = np.array(PT); has_pt = np.isfinite(PT).any()
    if has_pt: PT = np.nan_to_num(PT, nan=np.nanmin(PT[np.isfinite(PT)]))
    return dict(base=clean(np.hstack([np.array(CONV), np.array(CDYN), clean(np.array(SEQ))])),
                PT=PT.reshape(-1,1), has_pt=has_pt, y=y)


def head(kind):
    if kind == "logistic": return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if kind == "rf": return RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=0, n_jobs=4)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default=""); args = ap.parse_args()
    cells = {}
    for t in args.tags:
        c = load(t)
        if c is None: continue
        ds, m = parse_tag(t); cells[(ds, m)] = c
    by_model = defaultdict(dict)
    for (ds, m), c in cells.items(): by_model[m][ds] = c

    rows = []
    for model, dss in sorted(by_model.items()):
        have = [d for d in DATASETS if d in dss]
        if len(have) < 3: continue
        for tgt in have:
            src = [dss[d] for d in have if d != tgt]
            Xs = np.vstack([s["base"] for s in src]); ys = np.concatenate([s["y"] for s in src])
            Xt = dss[tgt]["base"]; yt = dss[tgt]["y"]
            r = {"model": model, "target": tgt, "n": int(len(yt))}
            for kind in ["logistic", "rf"]:
                clf = head(kind).fit(Xs, ys)
                r[kind] = roc_auc_score(yt, clf.predict_proba(Xt)[:, 1])
            # +ptrue logistic (only if all cells have ptrue)
            if all(s["has_pt"] for s in src) and dss[tgt]["has_pt"]:
                Xsp = np.hstack([Xs, np.vstack([s["PT"] for s in src])])
                Xtp = np.hstack([Xt, dss[tgt]["PT"]])
                clf = head("logistic").fit(Xsp, ys)
                r["logistic+ptrue"] = roc_auc_score(yt, clf.predict_proba(Xtp)[:, 1])
            rows.append(r)

    def macro(k):
        v = [r[k] for r in rows if k in r]; return np.mean(v), len(v)
    print("\n=== LODO transfer: head capacity (logistic vs RF) on SAME ChainUQ features ===")
    for k in ["logistic", "rf", "logistic+ptrue"]:
        mv, n = macro(k)
        wins = sum(1 for r in rows if k in r and "logistic" in r and r[k] > r["logistic"]) if k != "logistic" else 0
        extra = f"  wins-vs-logistic {wins}/{n}" if k != "logistic" else ""
        print(f"  {k:16s} macro {mv:.3f} (n={n}){extra}")
    # paired rf - logistic
    d = [r["rf"] - r["logistic"] for r in rows if "rf" in r]
    print(f"\n  RF - logistic (transfer): mean {np.mean(d):+.4f}, wins {sum(x>0 for x in d)}/{len(d)}, min {min(d):+.3f} max {max(d):+.3f}")
    dp = [r["logistic+ptrue"] - r["logistic"] for r in rows if "logistic+ptrue" in r]
    if dp: print(f"  +P(True) - logistic (transfer): mean {np.mean(dp):+.4f}, wins {sum(x>0 for x in dp)}/{len(dp)}")
    if args.out:
        json.dump({"rows": rows}, open(args.out, "w"), indent=2); print("saved", args.out)


if __name__ == "__main__":
    main()
