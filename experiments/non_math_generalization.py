"""P0-2 non-math generalization analysis (More_EXP.md Sec 2.2 P0-2).

Two layers, both judge-free (MC exact-match labels):
  1. in-domain non-math : train+eval ChainUQ (5-seed CV) on each non-math cell,
     vs same-cost baselines + SC@{2,4,8}. Proves ChainUQ doesn't depend on a math
     answer parser.
  2. math -> non-math transfer : head trained ONLY on the 31 math cells, applied
     zero-shot to each non-math cell. Stronger generalization evidence.

Only cells with >= --min-wrong errors enter aggregates (AUROC needs both classes;
several BBH cells saturate for strong models). Cells below threshold are reported
but flagged and excluded from macro — logged, not silently dropped.

Outputs non_math_indomain.json + math_to_nonmath.json.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from acd.env import EXP_ROOT
from acd import baselines as S
from acd import metrics as UQ

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


def sc(ans, k):
    a = [x for x in ans[:k] if x is not None]
    return Counter(a).most_common(1)[0][1] / len(a) if a else 0.0


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
    logp, selfc, pt, ans = [], [], [], []
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
        ans.append(sampans.get(rid, []))
        y.append(labs[rid])
    y = np.array(y)
    return dict(CONV=np.array(CONV), CDYN=np.array(CDYN), SEQ=clean(np.array(SEQ)),
                FINAL=np.array(FINAL), y=y, logp=np.array(logp), selfc=np.array(selfc),
                pt=np.array(pt), ans=ans)


def cq(c):
    return np.hstack([c["CONV"], c["CDYN"], c["SEQ"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonmath-tags", nargs="+", required=True)
    ap.add_argument("--math-tags", nargs="+", required=True, help="source math cells for transfer")
    ap.add_argument("--min-wrong", type=int, default=15)
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    # ---------- layer 1: in-domain non-math ----------
    indom = []
    print("\n=== P0-2 layer 1: IN-DOMAIN non-math (5-seed CV AUROC) ===")
    print(f"{'cell':26s}{'n':>5s}{'acc':>6s}{'wrong':>6s}{'logp':>6s}{'selfc':>6s}"
          f"{'C+F':>6s}{'ChainUQ':>8s}{'SC@8':>6s}")
    for tag in args.nonmath_tags:
        c = load(tag)
        if c is None:
            print(f"{tag:26s}  (missing cache)"); continue
        y = c["y"]; nwrong = int((1 - y).sum())
        if len(y) < 20 or y.sum() == 0 or nwrong == 0:
            print(f"{tag:26s}{len(y):5d}{y.mean():6.3f}{nwrong:6d}  (degenerate labels)")
            continue
        a = lambda s: roc_auc_score(y, clean(np.asarray(s)).ravel())
        row = {"tag": tag, "n": int(len(y)), "acc": float(y.mean()), "n_wrong": nwrong,
               "logp": a(c["logp"]), "self_certainty": a(c["selfc"]),
               "conv+final": roc_auc_score(y, oof(np.hstack([c["CONV"], c["FINAL"]]), y)),
               "chainuq": roc_auc_score(y, oof(cq(c), y))}
        if any(c["ans"]):
            for k in (2, 4, 8):
                row[f"sc@{k}"] = roc_auc_score(y, np.array([sc(x, k) for x in c["ans"]]))
        row["enough"] = nwrong >= args.min_wrong
        indom.append(row)
        flag = "" if row["enough"] else "  (< min-wrong, excl. from macro)"
        print(f"{tag:26s}{row['n']:5d}{row['acc']:6.3f}{nwrong:6d}{row['logp']:6.3f}"
              f"{row['self_certainty']:6.3f}{row['conv+final']:6.3f}{row['chainuq']:8.3f}"
              f"{row.get('sc@8', float('nan')):6.3f}{flag}")
    good = [r for r in indom if r["enough"]]
    if good:
        for m in ["logp", "self_certainty", "conv+final", "chainuq", "sc@8"]:
            vals = [r[m] for r in good if m in r]
            if vals:
                print(f"  macro {m:14s} {np.mean(vals):.3f}  (over {len(vals)} cells w/ >= {args.min_wrong} errors)")
        d = np.mean([r["chainuq"] - r["conv+final"] for r in good])
        beat = sum(1 for r in good if r["chainuq"] > r["conv+final"])
        print(f"  ChainUQ - CONV+FINAL macro {d:+.3f}; ChainUQ wins {beat}/{len(good)}")

    # ---------- layer 2: math -> non-math transfer ----------
    print("\n=== P0-2 layer 2: MATH -> NON-MATH transfer (zero-shot head) ===")
    srcs = [load(t) for t in args.math_tags]
    srcs = [s for s in srcs if s is not None and len(s["y"]) >= 20]
    FEATS = {"chainuq": cq, "conv+final": lambda c: np.hstack([c["CONV"], c["FINAL"]]),
             "seq-only": lambda c: c["SEQ"]}
    clfs = {}
    for fs, fn in FEATS.items():
        Xs = clean(np.vstack([fn(s) for s in srcs])); ys = np.concatenate([s["y"] for s in srcs])
        clfs[fs] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xs, ys)
    trans = []
    print(f"{'cell':26s}{'n':>5s}{'wrong':>6s}{'chainuq':>8s}{'C+F':>6s}{'seq':>6s}{'Δ':>7s}")
    for tag in args.nonmath_tags:
        c = load(tag)
        if c is None:
            continue
        y = c["y"]; nwrong = int((1 - y).sum())
        if len(y) < 20 or y.sum() == 0 or nwrong == 0:
            continue
        au = {fs: roc_auc_score(y, clfs[fs].predict_proba(clean(FEATS[fs](c)))[:, 1]) for fs in FEATS}
        base = max(au["conv+final"], au["seq-only"])
        row = {"tag": tag, "n": int(len(y)), "n_wrong": nwrong, "au": au,
               "delta": au["chainuq"] - base, "enough": nwrong >= args.min_wrong}
        trans.append(row)
        flag = "" if row["enough"] else "  (excl. macro)"
        print(f"{tag:26s}{row['n']:5d}{nwrong:6d}{au['chainuq']:8.3f}{au['conv+final']:6.3f}"
              f"{au['seq-only']:6.3f}{row['delta']:+7.3f}{flag}")
    goodt = [r for r in trans if r["enough"]]
    if goodt:
        print(f"  macro ChainUQ transfer {np.mean([r['au']['chainuq'] for r in goodt]):.3f} "
              f"| Δ vs best base {np.mean([r['delta'] for r in goodt]):+.3f} "
              f"| Δ>0 {sum(1 for r in goodt if r['delta']>0)}/{len(goodt)}")

    if args.out_prefix:
        json.dump({"rows": indom}, open(f"{args.out_prefix}_indomain.json", "w"), indent=2)
        json.dump({"rows": trans}, open(f"{args.out_prefix}_transfer.json", "w"), indent=2)
        print(f"\nsaved {args.out_prefix}_indomain.json + _transfer.json")


if __name__ == "__main__":
    main()
