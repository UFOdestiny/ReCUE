"""Multi-generation-seed variance table.

For each (model, dataset) with multiple generation seeds (base + _s2 + _s3), compute
AUROC of key methods per seed, then report mean +/- std across seeds. Uses cached
cdyn/labels/gen/sampans (fast).
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from collections import defaultdict, Counter

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from acd.env import EXP_ROOT
from acd import baselines as S

CVSEEDS = [2026, 7, 13]


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
    for s in CVSEEDS:
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        acc += o
    return acc / len(CVSEEDS)


def sc(ans, k):
    a = [x for x in ans[:k] if x is not None]
    return Counter(a).most_common(1)[0][1] / len(a) if a else 0.0


def eval_tag(tag):
    cdf = EXP_ROOT / "cdyn" / f"{tag}.json"
    lf = EXP_ROOT / "labels" / f"{tag}.json"
    gf = EXP_ROOT / "gen" / f"{tag}.json"
    saf = EXP_ROOT / "sampans" / f"{tag}.json"
    if not (cdf.exists() and lf.exists() and gf.exists()):
        return None
    cd = json.loads(cdf.read_text()); labs = json.loads(lf.read_text())
    gen = {g["id"]: g for g in json.loads(gf.read_text())}
    sa = json.loads(saf.read_text()) if saf.exists() else {}
    y, CONV, CDYN, SEQ, ans, ptrue = [], [], [], [], [], []
    ptf = EXP_ROOT / "ptrue" / f"{tag}_ptrue.json"
    pt = json.loads(ptf.read_text()) if ptf.exists() else None
    for i, c in cd.items():
        if i not in labs:
            continue
        CONV.append(c["conv"]); CDYN.append(c["cdyn"])
        g = gen.get(i, {})
        SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        ans.append(sa.get(i, [])); ptrue.append(pt.get(i, 0.5) if pt else np.nan)
        y.append(labs[i])
    y = np.array(y)
    if len(y) < 20 or y.sum() == 0 or (1 - y).sum() == 0:
        return None
    CONV = np.array(CONV); CDYN = np.array(CDYN); SEQ = clean(np.array(SEQ))
    out = {
        "ours": roc_auc_score(y, oof(np.hstack([CONV, CDYN, SEQ]), y)),
        "answer_conv": roc_auc_score(y, oof(CONV, y)),
        "logprob": roc_auc_score(y, clean(SEQ[:, 0]).ravel()),
        "sc@8": roc_auc_score(y, np.array([sc(a, 8) for a in ans])) if any(ans) else np.nan,
    }
    if pt is not None:
        out["p_true"] = roc_auc_score(y, clean(np.array(ptrue)).ravel())
    return out


def main():
    ap = argparse.ArgumentParser()
    # base pairs: "model|dataset" -> tries {ds}_{model}_k8, _s2, _s3
    ap.add_argument("--cells", nargs="+", required=True, help="dataset_model tags WITHOUT _k8")
    args = ap.parse_args()
    methods = ["logprob", "p_true", "answer_conv", "ours", "sc@8"]
    print(f"{'cell':26s}{'seeds':>6s}" + "".join(f"{m:>16s}" for m in methods))
    print("-" * (32 + 16 * len(methods)))
    for cell in args.cells:
        # seed variants
        variants = [f"{cell}_k8", f"{cell}_s2_k8", f"{cell}_s3_k8"]
        # also handle _s2 inserted before _k8 in tag naming: ds_model_s2_k8
        per = {m: [] for m in methods}
        nseed = 0
        for v in variants:
            r = eval_tag(v)
            if r is None:
                continue
            nseed += 1
            for m in methods:
                if m in r and np.isfinite(r[m]):
                    per[m].append(r[m])
        if nseed == 0:
            continue
        line = f"{cell:26s}{nseed:6d}"
        for m in methods:
            if per[m]:
                mu = np.mean(per[m]); sd = np.std(per[m])
                line += f"  {mu:.3f}±{sd:.3f}"
            else:
                line += f"{'--':>16s}"
        print(line)


if __name__ == "__main__":
    main()
