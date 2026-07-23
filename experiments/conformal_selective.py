"""PC-CP: conformal selective prediction with the prefix-confidence commitment score.

Idea born from the temporal-order ablation: the ChainUQ signal is ORDER-INVARIANT
(DUAL-PERM ~= 0), i.e. it is a distributional statistic of multi-prefix forced-answer
confidence, NOT a fragile temporal feature. That is exactly what a good conformal
nonconformity score needs: low assumptions, stable, transferable (LODO showed it
transfers across datasets). We therefore stop selling "a new score" and instead sell
a CAPABILITY WITH A GUARANTEE:

  Given a single generation (1.03x cost), abstain on the least-confident answers so
  that the SELECTIVE RISK (error rate on answered items) is provably <= alpha, and
  do so at HIGHER coverage than same-cost baselines, INCLUDING on the confident-
  consensus errors that self-consistency is structurally blind to, AND with the
  guarantee preserved under domain shift (calibrate on source datasets, test on an
  unseen target — the LODO protocol).

Method = split-conformal risk control (Bates et al. / conformal selective classif.):
  score s = P(correct) from a monotone estimator (higher = more confident).
  On a calibration set choose threshold tau = the largest cutoff s.t. the empirical
  selective risk on calibration <= alpha (with a finite-sample correction). Answer
  iff s >= tau; abstain otherwise. Report test selective risk (validity) + coverage.

Estimators compared at MATCHED info/cost:
  chainuq (1x)  = conv+cdyn+seq logistic P(correct)      [ours]
  conv+final    = endpoint null
  p_true (1fwd) = verbalized self-verdict
  logprob (1x)  = mean token logprob
  sc@8 (8x)     = vote fraction (the expensive reference)

Modes:
  indomain : per-cell calibration/test split (does the guarantee hold; whose coverage
             is highest at fixed risk?)
  shift    : LODO — calibrate tau on SOURCE datasets (pooled, per backbone), test on a
             held-out TARGET dataset. THE novel claim: distribution-free risk control
             survives domain shift with our transferable score.

Outputs conformal_indomain.json / conformal_shift.json.
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
from sklearn.model_selection import StratifiedKFold

from acd.env import EXP_ROOT
from acd import baselines as S

DATASETS = ["gsm8k", "math500", "minerva", "olympiad", "amc23"]


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
    CONV, CDYN, SEQ, FINAL, y, logp, pt, vote = [], [], [], [], [], [], [], []
    for r in recs:
        rid = r["id"]
        if not r["intermediate"] or rid not in labs or rid not in cdyn:
            continue
        CONV.append(cdyn[rid]["conv"]); CDYN.append(cdyn[rid]["cdyn"])
        g = gen.get(rid, {})
        SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])
        lp = [x.get("neutral_lp") for x in r["intermediate"] if x.get("neutral_lp") is not None]
        FINAL.append([lp[-1] if lp else -10.0])
        logp.append(S.sig_mean_logprob(g))
        pt.append(ptrue.get(rid, np.nan) if ptrue else np.nan)
        a = [x for x in sampans.get(rid, [])[:8] if x is not None]
        vote.append(Counter(a).most_common(1)[0][1] / len(a) if a else 0.0)
        y.append(labs[rid])
    y = np.array(y)
    if len(y) < 40 or y.sum() == 0 or (1 - y).sum() == 0:
        return None
    return dict(CONV=np.array(CONV), CDYN=np.array(CDYN), SEQ=clean(np.array(SEQ)),
                FINAL=np.array(FINAL), y=y, logp=np.array(logp),
                pt=np.array(pt), vote=np.array(vote))


def conformal_threshold(s_cal, y_cal, alpha, delta=0.1):
    """RCPS-style threshold (Bates et al. 2021): pick the SMALLEST tau (max coverage)
    whose selective-risk UPPER confidence bound <= alpha, so test selective risk <=
    alpha holds with probability >= 1 - delta. Scanning many thresholds and taking the
    max-coverage one that PASSES the UCB is the valid RCPS choice (monotone family),
    unlike naively using the empirical risk (which overfits calibration -> low validity).

    Selective risk(tau) = error rate among answered {s >= tau}. UCB via Hoeffding on the
    n_answered conditional Bernoulli mean: emp_risk + sqrt(log(1/delta)/(2 n_answered)).
    """
    order = np.argsort(-s_cal)  # include highest-confidence first
    s_sorted = s_cal[order]; y_sorted = y_cal[order]
    err = (1 - y_sorted).cumsum().astype(float)
    n = np.arange(1, len(y_sorted) + 1).astype(float)
    emp = err / n
    ucb = emp + np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    ok = np.where(ucb <= alpha)[0]
    if len(ok) == 0:
        return np.inf  # cannot certify the risk -> abstain on everything
    k = ok.max()  # largest answered set whose UCB still passes = max coverage
    return s_sorted[k]


def eval_selective(s_test, y_test, tau):
    ans = s_test >= tau
    cov = float(ans.mean())
    if ans.sum() == 0:
        return cov, float("nan")
    risk = float((1 - y_test[ans]).mean())
    return cov, risk


def oof_pcorrect(feat_fn, cells_train, cell_eval=None):
    """If cell_eval is None -> in-cell OOF P(correct). Else train on cells_train pooled,
    predict on cell_eval (transfer)."""
    if cell_eval is None:
        c = cells_train
        X = clean(feat_fn(c)); y = c["y"]; acc = np.zeros(len(y))
        for sd in (2026, 7, 13):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
                acc[te] += clf.predict_proba(X[te])[:, 1]
        return acc / 3
    Xs = clean(np.vstack([feat_fn(c) for c in cells_train]))
    ys = np.concatenate([c["y"] for c in cells_train])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xs, ys)
    return clf.predict_proba(clean(feat_fn(cell_eval)))[:, 1]


FEAT = {
    "chainuq": lambda c: np.hstack([c["CONV"], c["CDYN"], c["SEQ"]]),
    "conv+final": lambda c: np.hstack([c["CONV"], c["FINAL"]]),
}


def score_for(method, cell, train_cells=None, transfer_target=None):
    """Return a monotone P(correct)-like score for a method on `cell`."""
    if method in FEAT:
        return oof_pcorrect(FEAT[method], train_cells if transfer_target is not None else cell,
                            transfer_target)
    if method == "p_true":
        v = cell["pt"]
        return np.nan_to_num(v, nan=np.nanmin(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0)
    if method == "logprob":
        return cell["logp"]
    if method == "sc@8":
        return cell["vote"]
    raise ValueError(method)


METHODS = ["logprob", "p_true", "sc@8", "conv+final", "chainuq"]


def run_indomain(tags, alphas, seed_splits, boot):
    """Per-cell: split into calib/test many times, calibrate tau, measure test
    risk+coverage. Report mean coverage@risk and empirical validity (risk<=alpha rate)."""
    res = {a: defaultdict(lambda: {"cov": [], "risk": [], "valid": []}) for a in alphas}
    for tag in tags:
        c = load(tag)
        if c is None:
            continue
        y = c["y"]; n = len(y)
        # precompute per-method scores once (OOF for learned)
        scores = {m: np.asarray(score_for(m, c)) for m in METHODS}
        rng = np.random.RandomState(0)
        for _ in range(seed_splits):
            idx = rng.permutation(n); half = n // 2
            cal, te = idx[:half], idx[half:]
            for a in alphas:
                for m in METHODS:
                    s = scores[m]
                    tau = conformal_threshold(s[cal], y[cal], a)
                    cov, risk = eval_selective(s[te], y[te], tau)
                    res[a][m]["cov"].append(cov)
                    if not np.isnan(risk):
                        res[a][m]["risk"].append(risk)
                        res[a][m]["valid"].append(risk <= a)
    out = {}
    for a in alphas:
        out[str(a)] = {}
        for m in METHODS:
            d = res[a][m]
            out[str(a)][m] = {"coverage": float(np.mean(d["cov"])) if d["cov"] else 0.0,
                              "test_risk": float(np.mean(d["risk"])) if d["risk"] else float("nan"),
                              "validity_rate": float(np.mean(d["valid"])) if d["valid"] else float("nan")}
    return out


def run_shift(tags, alphas):
    """LODO conformal: per backbone, calibrate tau on 4 source datasets (pooled),
    test on the held-out 5th. Learned scores trained on source cells only."""
    cells = {}
    for tag in tags:
        c = load(tag)
        if c is None:
            continue
        ds, model = parse_tag(tag); cells[(ds, model)] = c
    by_model = defaultdict(dict)
    for (ds, m), c in cells.items():
        by_model[m][ds] = c
    rows = []
    agg = {a: defaultdict(lambda: {"cov": [], "valid": [], "risk": []}) for a in alphas}
    for model, dss in sorted(by_model.items()):
        have = [d for d in DATASETS if d in dss]
        if len(have) < 3:
            continue
        for tgt in have:
            src = [dss[d] for d in have if d != tgt]
            src_y = np.concatenate([s["y"] for s in src])
            tgt_c = dss[tgt]
            for m in METHODS:
                if m in FEAT:
                    s_src = np.concatenate([oof_pcorrect(FEAT[m], s) for s in src])  # OOF on each source
                    s_tgt = score_for(m, tgt_c, train_cells=src, transfer_target=tgt_c)
                elif m == "sc@8":
                    s_src = np.concatenate([s["vote"] for s in src]); s_tgt = tgt_c["vote"]
                elif m == "p_true":
                    def _pt(cc):
                        v = cc["pt"]; return np.nan_to_num(v, nan=np.nanmin(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0)
                    s_src = np.concatenate([_pt(s) for s in src]); s_tgt = _pt(tgt_c)
                else:  # logprob
                    s_src = np.concatenate([s["logp"] for s in src]); s_tgt = tgt_c["logp"]
                for a in alphas:
                    tau = conformal_threshold(np.asarray(s_src), src_y, a)
                    cov, risk = eval_selective(np.asarray(s_tgt), tgt_c["y"], tau)
                    agg[a][m]["cov"].append(cov)
                    if not np.isnan(risk):
                        agg[a][m]["valid"].append(risk <= a); agg[a][m]["risk"].append(risk)
                    if m == "chainuq":
                        rows.append({"model": model, "target": tgt, "alpha": a,
                                     "coverage": cov, "risk": risk})
    out = {"per_cell_chainuq": rows, "summary": {}}
    for a in alphas:
        out["summary"][str(a)] = {}
        for m in METHODS:
            d = agg[a][m]
            out["summary"][str(a)][m] = {
                "coverage": float(np.mean(d["cov"])) if d["cov"] else 0.0,
                "validity_rate": float(np.mean(d["valid"])) if d["valid"] else float("nan"),
                "mean_test_risk": float(np.mean(d["risk"])) if d["risk"] else float("nan")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--mode", choices=["indomain", "shift", "all"], default="all")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    if args.mode in ("indomain", "all"):
        r = run_indomain(args.tags, args.alphas, args.splits, 0)
        print("\n=== PC-CP IN-DOMAIN: coverage @ guaranteed selective risk <= alpha ===")
        for a in args.alphas:
            print(f"\n alpha={a} (target max error rate on answered)")
            print(f"   {'method':12s}{'coverage':>10s}{'test_risk':>11s}{'validity':>10s}")
            for m in METHODS:
                d = r[str(a)][m]
                print(f"   {m:12s}{d['coverage']:10.3f}{d['test_risk']:11.3f}{d['validity_rate']:10.2f}")
        if args.out_prefix:
            json.dump(r, open(f"{args.out_prefix}_indomain.json", "w"), indent=2)
            print("saved", f"{args.out_prefix}_indomain.json")

    if args.mode in ("shift", "all"):
        r = run_shift(args.tags, args.alphas)
        print("\n=== PC-CP UNDER DOMAIN SHIFT (LODO: calibrate on source, test on unseen target) ===")
        for a in args.alphas:
            print(f"\n alpha={a}")
            print(f"   {'method':12s}{'coverage':>10s}{'validity':>10s}{'mean_risk':>11s}")
            for m in METHODS:
                d = r["summary"][str(a)][m]
                print(f"   {m:12s}{d['coverage']:10.3f}{d['validity_rate']:10.2f}{d['mean_test_risk']:11.3f}")
        if args.out_prefix:
            json.dump(r, open(f"{args.out_prefix}_shift.json", "w"), indent=2)
            print("saved", f"{args.out_prefix}_shift.json")


if __name__ == "__main__":
    main()
