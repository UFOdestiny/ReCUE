"""P0-1 transfer: does the ChainUQ signal transfer across datasets/models, and
can a single global head replace per-cell heads? (More_EXP.md Sec 2.2)

Directly answers the reviewer attack "it's just per-cell supervised feature
engineering". A head is trained on SOURCE cells and evaluated on completely
unseen TARGET cells. Normalization/feature-selection/classifier are fit on
source only; dataset ID / accuracy / target statistics never enter the head.

Feature sets (all judge-free, from cached probes):
  ChainUQ    = CONV(6 identity) + CDYN(6 confidence-dynamics) + SEQ(2)   [reuses cdyn cache]
  CONV+FINAL = CONV(6) + final-probe confidence(1)      (endpoint null baseline)
  seq-only   = SEQ(2)                                    (mean_logprob, mean_entropy)

Modes:
  --mode lodo    per-backbone leave-one-DATASET-out (train 4 math ds -> test 5th)
  --mode lomo    leave-one-model-FAMILY-out (train other families -> test family)
                 + intra-family size transfer (Qwen3 4B/8B -> 14B)
  --mode global  ONE head over all training cells, evaluate on held-out problems
                 of every cell (group 5-fold by cell so target problems unseen);
                 also 'global+cellid' upper bound.

Metrics: macro AUROC over held-out domains, worst-domain AUROC, and
source->target degradation vs the in-domain per-cell head. Paired bootstrap CI
on the primary contrast ChainUQ_transfer - max(CONV+FINAL, seq-only)_transfer.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from acd.env import EXP_ROOT
from acd import baselines as S

DATASETS = ["gsm8k", "math500", "minerva", "olympiad", "amc23"]
# model key in tag -> family group for LOMO (sizes collapse within Qwen3)
FAMILY = {
    "qwen4b": "Qwen3", "qwen8b": "Qwen3", "qwen14b": "Qwen3",
    "qwen35_9b": "Qwen3.5", "phi4r": "Phi4", "ministral": "Ministral",
    "llama8b": "Llama",
}


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
    parts = tag.split("_")
    ds = parts[0]
    model = "_".join(parts[1:-1])  # drop trailing k8
    return ds, model


def load_cell(tag):
    """Return dict with per-example feature blocks + labels for a cell, or None."""
    cf = EXP_ROOT / "conf" / f"{tag}_conf.json"
    lf = EXP_ROOT / "labels" / f"{tag}.json"
    gf = EXP_ROOT / "gen" / f"{tag}.json"
    cdf = EXP_ROOT / "cdyn" / f"{tag}.json"
    if not (cf.exists() and lf.exists() and gf.exists() and cdf.exists()):
        return None
    recs = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
    cdyn = json.loads(cdf.read_text())
    gen = {g["id"]: g for g in json.loads(gf.read_text())}
    CONV, CDYN, SEQ, FINAL, y = [], [], [], [], []
    for r in recs:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in cdyn:
            continue
        CONV.append(cdyn[rid]["conv"]); CDYN.append(cdyn[rid]["cdyn"])
        g = gen.get(rid, {})
        SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        lp = [x.get("neutral_lp") for x in r["intermediate"] if x.get("neutral_lp") is not None]
        FINAL.append([lp[-1] if lp else -10.0])
        y.append(labs[rid])
    y = np.array(y)
    if len(y) < 20 or y.sum() == 0 or (1 - y).sum() == 0:
        return None
    return {"CONV": np.array(CONV), "CDYN": np.array(CDYN), "SEQ": clean(np.array(SEQ)),
            "FINAL": np.array(FINAL), "y": y}


FEATSETS = {
    "chainuq": lambda c: np.hstack([c["CONV"], c["CDYN"], c["SEQ"]]),
    "conv+final": lambda c: np.hstack([c["CONV"], c["FINAL"]]),
    "seq-only": lambda c: c["SEQ"],
}


def fit_transfer(src_cells, tgt_cell, fs):
    """Train on pooled source cells, score target. Returns AUROC on target."""
    Xs = clean(np.vstack([FEATSETS[fs](c) for c in src_cells]))
    ys = np.concatenate([c["y"] for c in src_cells])
    Xt = clean(FEATSETS[fs](tgt_cell)); yt = tgt_cell["y"]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xs, ys)
    s = clf.predict_proba(Xt)[:, 1]
    return roc_auc_score(yt, s), s


def indomain(cell, fs, seeds=(2026, 7, 13)):
    X = clean(FEATSETS[fs](cell)); y = cell["y"]; acc = np.zeros(len(y))
    for sd in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
            acc[te] += clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, acc / len(seeds))


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


def run_lodo(cells, boot):
    """Per-backbone leave-one-dataset-out. cells keyed by (ds, model)."""
    by_model = defaultdict(dict)
    for (ds, model), c in cells.items():
        by_model[model][ds] = c
    rows = []
    for model, dss in sorted(by_model.items()):
        have = [d for d in DATASETS if d in dss]
        if len(have) < 3:   # need enough source datasets; skip llama (math500 only)
            continue
        for tgt in have:
            src = [dss[d] for d in have if d != tgt]
            au = {}; scores = {}
            for fs in FEATSETS:
                au[fs], scores[fs] = fit_transfer(src, dss[tgt], fs)
            base = "conv+final" if au["conv+final"] >= au["seq-only"] else "seq-only"
            md, lo, hi, p = boot_p(dss[tgt]["y"], scores["chainuq"], scores[base], n=boot)
            ind = indomain(dss[tgt], "chainuq")
            rows.append({"model": model, "target": tgt, "n": int(len(dss[tgt]["y"])),
                         "au": au, "vs_base": base, "delta": md, "ci": [lo, hi], "p": p,
                         "indomain": ind, "degradation": ind - au["chainuq"]})
    return rows


def run_lomo(cells, boot):
    by_fam = defaultdict(list)
    cell_fam = {}
    for (ds, model), c in cells.items():
        fam = FAMILY.get(model, model)
        by_fam[fam].append(c); cell_fam[(ds, model)] = fam
    rows = []
    fams = [f for f in by_fam if len(by_fam[f]) >= 2 and f != "Llama"]
    for held in fams:
        src = [c for f in by_fam for c in by_fam[f] if f != held]
        # evaluate pooled over the held-out family's cells
        tgt_cells = by_fam[held]
        for fs in FEATSETS:
            pass
        # build pooled target scores per featureset
        au = {}; ys = np.concatenate([c["y"] for c in tgt_cells]); sc = {}
        for fs in FEATSETS:
            Xs = clean(np.vstack([FEATSETS[fs](c) for c in src]))
            ysrc = np.concatenate([c["y"] for c in src])
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xs, ysrc)
            s = np.concatenate([clf.predict_proba(clean(FEATSETS[fs](c)))[:, 1] for c in tgt_cells])
            sc[fs] = s; au[fs] = roc_auc_score(ys, s)
        base = "conv+final" if au["conv+final"] >= au["seq-only"] else "seq-only"
        md, lo, hi, p = boot_p(ys, sc["chainuq"], sc[base], n=boot)
        rows.append({"held_family": held, "n": int(len(ys)), "au": au,
                     "vs_base": base, "delta": md, "ci": [lo, hi], "p": p})
    return rows


def run_size(cells, boot):
    """Intra-Qwen3 size transfer: train {4B,8B} -> test 14B (per dataset pooled)."""
    train_models = ["qwen4b", "qwen8b"]; tgt_model = "qwen14b"
    src = [c for (ds, m), c in cells.items() if m in train_models]
    tgt = [c for (ds, m), c in cells.items() if m == tgt_model]
    if not src or not tgt:
        return []
    ys = np.concatenate([c["y"] for c in tgt]); au = {}; sc = {}
    for fs in FEATSETS:
        Xs = clean(np.vstack([FEATSETS[fs](c) for c in src]))
        ysrc = np.concatenate([c["y"] for c in src])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xs, ysrc)
        s = np.concatenate([clf.predict_proba(clean(FEATSETS[fs](c)))[:, 1] for c in tgt])
        sc[fs] = s; au[fs] = roc_auc_score(ys, s)
    base = "conv+final" if au["conv+final"] >= au["seq-only"] else "seq-only"
    md, lo, hi, p = boot_p(ys, sc["chainuq"], sc[base], n=boot)
    return [{"transfer": "Qwen3-4B/8B -> 14B", "n": int(len(ys)), "au": au,
             "vs_base": base, "delta": md, "ci": [lo, hi], "p": p}]


def run_global(cells, boot):
    """One head across ALL cells; group 5-fold by cell so eval problems unseen
    at cell level is NOT the goal (that's LODO). Here folds are within-pool but
    we hold out whole cells in rotation to keep target problems unseen by their
    OWN training rows only via stratified split. We report the honest 'global,
    no cell-id' head evaluated out-of-fold, plus a 'global+cellid' upper bound."""
    tags = sorted(cells.keys())
    # assemble pooled matrices with cell one-hot for the upper bound
    fs = "chainuq"
    blocks = [FEATSETS[fs](cells[t]) for t in tags]
    ys = [cells[t]["y"] for t in tags]
    lens = [len(y) for y in ys]
    X = clean(np.vstack(blocks)); y = np.concatenate(ys)
    cellidx = np.concatenate([[i] * l for i, l in enumerate(lens)])
    onehot = np.eye(len(tags))[cellidx]
    # LOCO: leave-one-cell-out global head (train on all OTHER cells)
    per_cell_au = {}; per_cell_scores = {}; ci_au = {}
    for i, t in enumerate(tags):
        tr = cellidx != i; teI = cellidx == i
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        s = clf.predict_proba(X[teI])[:, 1]
        yt = y[teI]
        if yt.sum() == 0 or (1 - yt).sum() == 0:
            continue
        tk = f"{t[0]}_{t[1]}" if isinstance(t, tuple) else t
        per_cell_au[tk] = roc_auc_score(yt, s); per_cell_scores[tk] = (yt, s)
    # global+cellid upper bound: standard grouped 5-fold with one-hot appended
    Xh = np.hstack([X, onehot])
    acc = np.zeros(len(y))
    for sd in (2026, 7, 13):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(Xh, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xh[tr], y[tr])
            acc[te] += clf.predict_proba(Xh[te])[:, 1]
    acc /= 3
    ub_au = {}
    off = 0
    for t, l in zip(tags, lens):
        yt = y[off:off + l]; st = acc[off:off + l]; off += l
        if yt.sum() and (1 - yt).sum():
            tk = f"{t[0]}_{t[1]}" if isinstance(t, tuple) else t
            ub_au[tk] = roc_auc_score(yt, st)
    return {"loco_global": per_cell_au, "global_cellid_ub": ub_au}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--mode", choices=["lodo", "lomo", "global", "all"], default="all")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    cells = {}
    for tag in args.tags:
        c = load_cell(tag)
        if c is None:
            continue
        ds, model = parse_tag(tag)
        cells[(ds, model)] = c
    print(f"[transfer] loaded {len(cells)} cells")

    def dump(obj, name):
        if args.out_prefix:
            path = f"{args.out_prefix}_{name}.json"
            json.dump(obj, open(path, "w"), indent=2); print("saved", path)

    if args.mode in ("lodo", "all"):
        rows = run_lodo(cells, args.boot)
        print("\n=== LODO (per-backbone leave-one-dataset-out) ===")
        print(f"{'model':12s}{'target':10s}{'n':>5s}{'chainuq':>9s}{'C+F':>7s}{'seq':>7s}"
              f"{'Δvbase':>8s}{'p':>7s}{'indom':>7s}{'degr':>7s}")
        for r in rows:
            star = "*" if r["p"] < 0.05 else " "
            print(f"{r['model']:12s}{r['target']:10s}{r['n']:5d}{r['au']['chainuq']:9.3f}"
                  f"{r['au']['conv+final']:7.3f}{r['au']['seq-only']:7.3f}{r['delta']:+8.3f}{r['p']:6.3f}{star}"
                  f"{r['indomain']:7.3f}{r['degradation']:+7.3f}")
        if rows:
            macro = np.mean([r["au"]["chainuq"] for r in rows])
            worst = np.min([r["au"]["chainuq"] for r in rows])
            pos = sum(1 for r in rows if r["delta"] > 0); sig = sum(1 for r in rows if r["p"] < 0.05)
            print(f"  MACRO chainuq {macro:.3f} | worst-domain {worst:.3f} | "
                  f"Δ>0 {pos}/{len(rows)} | sig {sig}/{len(rows)}")
        dump(rows, "lodo")

    if args.mode in ("lomo", "all"):
        rows = run_lomo(cells, args.boot); size = run_size(cells, args.boot)
        print("\n=== LOMO (leave-one-model-family-out) ===")
        print(f"{'held family':14s}{'n':>6s}{'chainuq':>9s}{'C+F':>7s}{'seq':>7s}{'Δvbase':>8s}{'p':>7s}")
        for r in rows:
            star = "*" if r["p"] < 0.05 else " "
            print(f"{r['held_family']:14s}{r['n']:6d}{r['au']['chainuq']:9.3f}{r['au']['conv+final']:7.3f}"
                  f"{r['au']['seq-only']:7.3f}{r['delta']:+8.3f}{r['p']:6.3f}{star}")
        if rows:
            print(f"  MACRO chainuq {np.mean([r['au']['chainuq'] for r in rows]):.3f} | "
                  f"Δ>0 {sum(1 for r in rows if r['delta']>0)}/{len(rows)}")
        for r in size:
            print(f"  size-transfer {r['transfer']}: chainuq {r['au']['chainuq']:.3f} "
                  f"(C+F {r['au']['conv+final']:.3f}) Δ {r['delta']:+.3f} p={r['p']:.3f}")
        dump({"lomo": rows, "size": size}, "lomo")

    if args.mode in ("global", "all"):
        g = run_global(cells, args.boot)
        loco = g["loco_global"]; ub = g["global_cellid_ub"]
        print("\n=== GLOBAL unified head ===")
        print(f"  leave-one-cell-out global head (no cell-id): macro {np.mean(list(loco.values())):.3f} "
              f"over {len(loco)} cells, worst {min(loco.values()):.3f}")
        print(f"  global + cell-id (upper bound):              macro {np.mean(list(ub.values())):.3f}")
        dump(g, "global")


if __name__ == "__main__":
    main()
