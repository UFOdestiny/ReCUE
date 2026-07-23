"""P0-3 decisive temporal-ORDER ablation (More_EXP.md Sec 2.3 / Block A).

Question: is ChainUQ's signal a genuine *temporal trajectory*, or just an
unordered multiset of prefix-probe confidences? We hold the marginal
information (the confidence multiset + identity counts) fixed and destroy ONLY
the order, then test whether performance drops.

Feature decomposition (from conf caches: per-cut `neutral` identity + `neutral_lp`):
  BAG (order-INVARIANT):
    confidence: mean, std, min, max, median, q25, q75
    identity  : agree_frac (mode share), n_distinct, none_frac, id_entropy
  ORD (order-SENSITIVE, added on top of BAG):
    confidence: slope, early_mean(1st third), late_gap(last-first third),
                area(trapz), n_rises, lag1_autocorr, argmax_pos, argmin_pos
    identity  : conv_frac, final_stable_run, flip_rate, first_agree_pos

Configs (5-seed CV AUROC; PERM/REVERSE add per-example sequence transforms):
  1. FINAL            final-probe confidence only
  2. CONV             identity-convergence bag+order (prior-art content)
  3. C+F              CONV + FINAL  (endpoint null hypothesis)
  4. BAG              order-invariant confidence+identity stats (multiset)
  5. PERM             BAG + ORD computed on RANDOMLY PERMUTED sequences,
                      train-permuted -> test-permuted, averaged over --perm-seeds
  6. REVERSE          BAG + ORD computed on time-REVERSED sequences
  7. ORD-CONF         confidence trajectory only (bag+order, no identity)
  8. ORD-ID           identity trajectory only
  9. DUAL             ordered dual trajectory (conf + identity, bag+order)
 10. FULL             DUAL + sequence features (mean_logprob, mean_entropy)

Primary contrasts (paired problem-level bootstrap CI):
  * DUAL - BAG        (does order add over the multiset?)
  * DUAL - PERM       (is the ORDER, not the retrained capacity, what helps?)
  * FULL - C+F        (full method over endpoint null)

Go/No-go: DUAL must beat BAG and PERM stably; else rename to
"multi-prefix commitment evidence" rather than "temporal dynamics".
"""
from __future__ import annotations

import argparse
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
_trapz = getattr(np, "trapezoid", np.trapz)

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
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


def oof(X, y, seeds=SEEDS):
    X = clean(X); acc = np.zeros(len(y))
    for s in seeds:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        acc += o
    return acc / len(seeds)


def _ids(neu):
    ids, reps = [], []
    for a in neu:
        if a is None:
            ids.append(-1); continue
        f = next((k for k, r in enumerate(reps) if _eq(a, r)), None)
        if f is None:
            reps.append(a); f = len(reps) - 1
        ids.append(f)
    return ids


def conf_bag(lpv):
    """order-INVARIANT confidence stats."""
    if not lpv:
        return [-10.0] * 7
    a = np.asarray(lpv, float)
    return [float(a.mean()), float(a.std() if len(a) > 1 else 0.0), float(a.min()),
            float(a.max()), float(np.median(a)),
            float(np.percentile(a, 25)), float(np.percentile(a, 75))]


def conf_ord(lpv):
    """order-SENSITIVE confidence trajectory features."""
    n = len(lpv)
    if n == 0:
        return [0.0] * 8
    a = np.asarray(lpv, float)
    slope = float(np.polyfit(np.arange(n), a, 1)[0]) if n >= 2 else 0.0
    third = max(1, n // 3)
    early = float(a[:third].mean())
    late_gap = float(a[-third:].mean() - early)
    area = float(_trapz(a) / max(n - 1, 1))
    nrise = sum(1 for i in range(1, n) if a[i] > a[i - 1]) / max(1, n - 1)
    if n >= 2 and a.std() > 1e-9:
        ac = float(np.corrcoef(a[:-1], a[1:])[0, 1])
        if not np.isfinite(ac):
            ac = 0.0
    else:
        ac = 0.0
    argmax_pos = float(np.argmax(a) / max(n - 1, 1))
    argmin_pos = float(np.argmin(a) / max(n - 1, 1))
    return [slope, early, late_gap, area, nrise, ac, argmax_pos, argmin_pos]


def id_bag(ids):
    """order-INVARIANT identity stats."""
    n = len(ids)
    valid = [i for i in ids if i != -1]
    from collections import Counter
    c = Counter(valid)
    agree = (c.most_common(1)[0][1] / len(valid)) if valid else 0.0
    ndist = float(len(set(valid))) if valid else 1.0
    none_frac = float(np.mean([1.0 if i == -1 else 0.0 for i in ids])) if n else 1.0
    tot = sum(c.values())
    ent = -sum((v / tot) * np.log(v / tot) for v in c.values()) if tot else 3.0
    return [float(agree), ndist, none_frac, float(ent)]


def id_ord(ids):
    """order-SENSITIVE identity trajectory features (vs the FINAL id)."""
    n = len(ids)
    if n == 0:
        return [0.0, 0.0, 1.0, 1.0]
    final = ids[-1]
    agree = [1.0 if (i == final and final != -1) else 0.0 for i in ids]
    run = 0
    for i in range(n - 1, -1, -1):
        if agree[i] == 1.0:
            run += 1
        else:
            break
    fst = run / n
    conv = 1.0
    for i in range(n):
        if all(agree[j] == 1.0 for j in range(i, n)):
            conv = (i + 1) / n; break
    flip = sum(1 for i in range(1, n) if ids[i] != ids[i - 1]) / max(1, n - 1)
    first_agree = 1.0
    for i in range(n):
        if agree[i] == 1.0:
            first_agree = (i + 1) / n; break
    return [conv, fst, flip, first_agree]


def build(recs, labs):
    """Return per-example raw sequences + labels + seq features."""
    LP, ID, FINAL, SEQ, y = [], [], [], [], []
    for r in recs:
        if not r["intermediate"] or r["id"] not in labs:
            continue
        neu = [x["neutral"] for x in r["intermediate"]]
        lp = [x.get("neutral_lp") for x in r["intermediate"]]
        lpv = [v for v in lp if v is not None]
        if not lpv:
            continue
        LP.append(lpv)
        ID.append(_ids(neu))
        FINAL.append([lpv[-1]])
        y.append(labs[r["id"]])
        SEQ.append(None)  # filled by caller w/ gen
    return LP, ID, FINAL, y


def feats_from_seq(lpv, ids, order=True):
    cb = conf_bag(lpv); ib = id_bag(ids)
    if not order:
        return cb + ib
    return cb + conf_ord(lpv) + ib + id_ord(ids)


def matrix(LP, ID, order=True, transform=None, rng=None):
    """Build feature matrix. transform in {None,'perm','reverse'}."""
    rows = []
    for lpv, ids in zip(LP, ID):
        if transform == "perm":
            p = rng.permutation(len(lpv))
            lpv2 = [lpv[i] for i in p]
            # permute identity seq with its own draw (same length assumption)
            q = rng.permutation(len(ids))
            ids2 = [ids[i] for i in q]
        elif transform == "reverse":
            lpv2 = lpv[::-1]; ids2 = ids[::-1]
        else:
            lpv2, ids2 = lpv, ids
        rows.append(feats_from_seq(lpv2, ids2, order=order))
    return np.array(rows)


def conf_only(LP, order=True, transform=None, rng=None):
    rows = []
    for lpv in LP:
        if transform == "perm":
            lpv2 = [lpv[i] for i in rng.permutation(len(lpv))]
        elif transform == "reverse":
            lpv2 = lpv[::-1]
        else:
            lpv2 = lpv
        rows.append(conf_bag(lpv2) + (conf_ord(lpv2) if order else []))
    return np.array(rows)


def id_only(ID, order=True):
    return np.array([id_bag(ids) + (id_ord(ids) if order else []) for ids in ID])


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
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d <= 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--perm-seeds", type=int, default=10)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    CFG = ["FINAL", "CONV", "C+F", "BAG", "PERM", "REVERSE",
           "ORD-CONF", "ORD-ID", "DUAL", "FULL"]
    agg = {k: [] for k in CFG}
    dc = {"DUAL-BAG": [], "DUAL-PERM": [], "FULL-C+F": []}
    sig = {"DUAL-BAG": 0, "DUAL-PERM": 0, "FULL-C+F": 0}
    results = []

    for tag in args.tags:
        cf = EXP_ROOT / "conf" / f"{tag}_conf.json"
        lf = EXP_ROOT / "labels" / f"{tag}.json"
        gf = EXP_ROOT / "gen" / f"{tag}.json"
        if not (cf.exists() and lf.exists() and gf.exists()):
            continue
        recs = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
        gen = {g["id"]: g for g in json.loads(gf.read_text())}
        LP, ID, FINAL, SEQ, y = [], [], [], [], []
        for r in recs:
            if not r["intermediate"] or r["id"] not in labs:
                continue
            neu = [x["neutral"] for x in r["intermediate"]]
            lp = [x.get("neutral_lp") for x in r["intermediate"]]
            lpv = [v for v in lp if v is not None]
            if not lpv:
                continue
            LP.append(lpv); ID.append(_ids(neu)); FINAL.append([lpv[-1]])
            g = gen.get(r["id"], {})
            SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
            y.append(labs[r["id"]])
        y = np.array(y)
        if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0:
            continue
        FINAL = np.array(FINAL); SEQ = clean(np.array(SEQ))

        s = {}
        s["FINAL"] = oof(FINAL, y)
        conv_mat = id_only(ID, order=True)
        s["CONV"] = oof(conv_mat, y)
        s["C+F"] = oof(np.hstack([conv_mat, FINAL]), y)
        s["BAG"] = oof(matrix(LP, ID, order=False), y)
        s["REVERSE"] = oof(matrix(LP, ID, order=True, transform="reverse"), y)
        s["ORD-CONF"] = oof(conf_only(LP, order=True), y)
        s["ORD-ID"] = oof(id_only(ID, order=True), y)
        dual = matrix(LP, ID, order=True)
        s["DUAL"] = oof(dual, y)
        s["FULL"] = oof(np.hstack([dual, SEQ]), y)
        # PERM: average AUROC over perm seeds (train-perm -> test-perm each seed)
        perm_scores = []
        for ps in range(args.perm_seeds):
            rng = np.random.RandomState(1000 + ps)
            Xp = matrix(LP, ID, order=True, transform="perm", rng=rng)
            perm_scores.append(oof(Xp, y, seeds=SEEDS[:3]))
        s["PERM"] = np.mean(perm_scores, axis=0)

        a = lambda z: roc_auc_score(y, z)
        au = {k: a(v) for k, v in s.items()}
        for k in CFG:
            agg[k].append(au[k])
        # contrasts
        for name, (A, B) in {"DUAL-BAG": ("DUAL", "BAG"),
                             "DUAL-PERM": ("DUAL", "PERM"),
                             "FULL-C+F": ("FULL", "C+F")}.items():
            md, lo, hi, p = boot_p(y, s[A], s[B], n=args.boot)
            dc[name].append(md); sig[name] += (p < 0.05)
        results.append({"tag": tag, "n": int(len(y)), "au": au})

    # ---- report ----
    print(f"\n{'tag':22s}" + "".join(f"{k:>8s}" for k in CFG))
    print("-" * (22 + 8 * len(CFG)))
    for r in results:
        print(f"{r['tag']:22s}" + "".join(f"{r['au'][k]:8.3f}" for k in CFG))
    print("-" * (22 + 8 * len(CFG)))
    print(f"{'MACRO':22s}" + "".join(f"{np.mean(agg[k]):8.3f}" for k in CFG))
    print()
    N = len(results)
    for name in ["DUAL-BAG", "DUAL-PERM", "FULL-C+F"]:
        print(f"{name:12s} mean Δ {np.mean(dc[name]):+.4f}  significant {sig[name]}/{N} cells")
    print("\nGo/No-go: DUAL must beat BAG and PERM. "
          f"DUAL macro {np.mean(agg['DUAL']):.3f} | BAG {np.mean(agg['BAG']):.3f} | "
          f"PERM {np.mean(agg['PERM']):.3f} | REVERSE {np.mean(agg['REVERSE']):.3f}")
    if args.out:
        json.dump({"config": CFG, "results": results,
                   "macro": {k: float(np.mean(agg[k])) for k in CFG},
                   "contrasts": {k: {"mean": float(np.mean(dc[k])), "sig": sig[k], "n": N}
                                 for k in dc}},
                  open(args.out, "w"), indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
