"""Reviewer 3.6 -- a STRONG supervised uncertainty-trajectory baseline.

Reviewer asks for a baseline in the spirit of "Uncertainty Trace Profiles" / token-
level entropy-trajectory methods, fed to the SAME logistic head as ReCUE, so the
comparison isolates the OBSERVATION (forced multi-prefix answer commitment) from the
classifier. This baseline uses ONLY the token-level logprob/entropy trajectory of the
single primary generation (no forced-answer probes at all):

  UTP features (per trace, from chosen_logprobs + topk_logprobs):
    - 8-bin mean token logprob (trajectory shape)
    - 8-bin mean token entropy
    - global mean/min/std logprob, mean/max entropy
    - slope & linearity (R^2) of logprob and entropy over token position
    - last-window (answer-region) mean logprob and entropy
    - fraction of low-confidence tokens (logprob < -2)
  => ~30-dim token-uncertainty trajectory descriptor.

Compared (macro AUROC, same 5-seed CV, 31 cells) against:
  mean_logprob, self_certainty, deepconf_bottom (existing 1x baselines) and ReCUE.
This directly answers "you only compared to weak 1x baselines; implement a supervised
trajectory profile."
"""
from __future__ import annotations

import argparse, json
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

from recue.env import EXP_ROOT
from recue import baselines as S

SEEDS = [2026, 7, 13, 42, 100]
NBIN = 8


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
        o = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            o[te] = c.predict_proba(X[te])[:, 1]
        acc += o
    return acc / len(SEEDS)


def hier_boot(cell_data, n=2000, seed=0):
    """Cell-equal hierarchical bootstrap for paired AUROC differences."""
    rng = np.random.RandomState(seed)
    draws = []
    for _ in range(n):
        cells = rng.choice(len(cell_data), len(cell_data), replace=True)
        diffs = []
        for ci in cells:
            y, score_a, score_b = cell_data[ci]
            pos = np.where(y == 1)[0]
            neg = np.where(y == 0)[0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            idx = np.concatenate([
                rng.choice(pos, len(pos), replace=True),
                rng.choice(neg, len(neg), replace=True),
            ])
            yy = y[idx]
            diffs.append(
                roc_auc_score(yy, score_a[idx])
                - roc_auc_score(yy, score_b[idx])
            )
        if diffs:
            draws.append(np.mean(diffs))
    draws = np.asarray(draws)
    return [
        float(np.mean(draws)),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
        float(np.mean(draws <= 0)),
    ]


def _entropy_traj(topk):
    ents = []
    for vals in topk:
        if not vals: continue
        p = np.exp(np.array(vals)); p = p / max(p.sum(), 1e-12)
        ents.append(-float(np.sum(p * np.log(p + 1e-12))))
    return np.array(ents)


def _binned(arr, nbin):
    if len(arr) == 0:
        return [0.0] * nbin
    idx = np.linspace(0, len(arr), nbin + 1).astype(int)
    out = []
    for i in range(nbin):
        seg = arr[idx[i]:idx[i + 1]]
        out.append(float(np.mean(seg)) if len(seg) else float(arr[min(idx[i], len(arr) - 1)]))
    return out


def _slope_r2(arr):
    n = len(arr)
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n)
    b1, b0 = np.polyfit(x, arr, 1)
    pred = b1 * x + b0
    ss_res = np.sum((arr - pred) ** 2); ss_tot = np.sum((arr - np.mean(arr)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(b1), float(r2)


def utp_features(g):
    lp = np.array(g.get("chosen_logprobs") or [], dtype=float)
    ent = _entropy_traj(g.get("topk_logprobs") or [])
    if len(lp) == 0:
        return [0.0] * (2 * NBIN + 11)
    feats = []
    feats += _binned(lp, NBIN)
    feats += _binned(ent, NBIN) if len(ent) else [0.0] * NBIN
    feats += [float(np.mean(lp)), float(np.min(lp)), float(np.std(lp))]
    feats += [float(np.mean(ent)) if len(ent) else 0.0, float(np.max(ent)) if len(ent) else 0.0]
    s_lp, r2_lp = _slope_r2(lp); s_e, r2_e = _slope_r2(ent) if len(ent) else (0.0, 0.0)
    feats += [s_lp, r2_lp, s_e, r2_e]
    feats += [float(np.mean(lp[-30:]))]                 # answer-region logprob
    feats += [float(np.mean(lp < -2.0))]                # low-confidence token frac
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    macro = defaultdict(list)
    rows = []
    full_vs_utp = []
    full_vs_active = []
    active_sc_vs_sc = []
    full_sc_vs_sc = []
    for tag in args.tags:
        gf = EXP_ROOT / "gen" / f"{tag}.json"
        lf = EXP_ROOT / "labels" / f"{tag}.json"
        cdf = EXP_ROOT / "cdyn" / f"{tag}.json"
        cf = EXP_ROOT / "conf" / f"{tag}_conf.json"
        saf = EXP_ROOT / "sampans" / f"{tag}.json"
        if not (gf.exists() and lf.exists() and cdf.exists() and cf.exists()):
            continue
        gen = {g["id"]: g for g in json.loads(gf.read_text())}
        labs = json.loads(lf.read_text()); cdyn = json.loads(cdf.read_text())
        conf = json.loads(cf.read_text())
        sampans = json.loads(saf.read_text()) if saf.exists() else {}
        y, UTP, LOGP, SELFC, DC, CONV, CDYN, SEQ, VOTE, ENT = (
            [], [], [], [], [], [], [], [], [], []
        )
        for r in conf:
            rid = r["id"]
            if not r["intermediate"] or rid not in labs or rid not in cdyn or rid not in gen:
                continue
            g = gen[rid]
            UTP.append(utp_features(g))
            LOGP.append(S.sig_mean_logprob(g)); SELFC.append(S.sig_self_certainty(g))
            DC.append(S.sig_deepconf_bottom(g))
            CONV.append(cdyn[rid]["conv"]); CDYN.append(cdyn[rid]["cdyn"])
            SEQ.append([S.sig_mean_logprob(g), S.sig_mean_entropy(g)])   # paper's SEQ block
            answers = [a for a in sampans.get(rid, [])[:8] if a is not None]
            counts = Counter(answers)
            total = sum(counts.values())
            VOTE.append(counts.most_common(1)[0][1] / total if total else 0.0)
            ENT.append(
                -sum((v / total) * np.log(v / total) for v in counts.values())
                if total else 0.0
            )
            y.append(labs[rid])
        y = np.array(y)
        if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0:
            continue
        CONV = np.array(CONV); CDYN = np.array(CDYN); SEQ = clean(np.array(SEQ)); UTP = np.array(UTP)
        base = np.hstack([CONV, CDYN, SEQ])                 # ReCUE-Active
        full = np.hstack([base, UTP])                       # full ReCUE
        a = lambda v: roc_auc_score(y, clean(np.asarray(v)).ravel())
        au = {}
        au["mean_logprob"] = a(LOGP); au["self_certainty"] = a(SELFC); au["deepconf"] = a(DC)
        score_utp = oof(UTP, y)
        score_active = oof(base, y)
        score_full = oof(full, y)
        au["utp_supervised"] = a(score_utp)
        au["recue_active"] = a(score_active)
        au["recue"] = a(score_full)
        full_vs_utp.append((y, score_full, score_utp))
        full_vs_active.append((y, score_full, score_active))

        vote = np.asarray(VOTE).reshape(-1, 1)
        ent = np.asarray(ENT).reshape(-1, 1)
        score_sc = vote.ravel()
        score_active_sc = oof(np.hstack([base, vote, ent]), y)
        score_full_sc = oof(np.hstack([full, vote, ent]), y)
        au["sc8"] = a(score_sc)
        au["recue_active+sc8"] = a(score_active_sc)
        au["recue+sc8"] = a(score_full_sc)
        active_sc_vs_sc.append((y, score_active_sc, score_sc))
        full_sc_vs_sc.append((y, score_full_sc, score_sc))
        rows.append((tag, len(y), au["mean_logprob"], au["self_certainty"], au["deepconf"],
                     au["utp_supervised"], au["recue_active"], au["recue"],
                     au["sc8"], au["recue_active+sc8"], au["recue+sc8"]))
        for k, v in au.items(): macro[k].append(v)

    print(f"\n{'tag':22s}{'n':>5s}{'mlogp':>7s}{'selfc':>7s}{'dconf':>7s}"
          f"{'UTP*':>7s}{'Active':>8s}{'Full':>8s}{'SC8':>7s}{'A+SC':>7s}{'F+SC':>7s}")
    print("-" * 97)
    for (tag, n, ml, sc, dc, utp, active, full_, sc8, asc, fsc) in rows:
        print(f"{tag:22s}{n:5d}{ml:7.3f}{sc:7.3f}{dc:7.3f}{utp:7.3f}"
              f"{active:8.3f}{full_:8.3f}{sc8:7.3f}{asc:7.3f}{fsc:7.3f}")
    print("-" * 97)
    print(f"{'MACRO':22s}{'':>5s}{np.mean(macro['mean_logprob']):7.3f}{np.mean(macro['self_certainty']):7.3f}"
          f"{np.mean(macro['deepconf']):7.3f}{np.mean(macro['utp_supervised']):7.3f}"
          f"{np.mean(macro['recue_active']):8.3f}{np.mean(macro['recue']):8.3f}"
          f"{np.mean(macro['sc8']):7.3f}{np.mean(macro['recue_active+sc8']):7.3f}"
          f"{np.mean(macro['recue+sc8']):7.3f}")
    print(f"\nUTP-supervised = strong learned token-trajectory baseline (same head, ~27 feats).")
    contrasts = {
        "recue_minus_utp": hier_boot(full_vs_utp),
        "recue_minus_active": hier_boot(full_vs_active),
        "active_sc8_minus_sc8": hier_boot(active_sc_vs_sc),
        "recue_sc8_minus_sc8": hier_boot(full_sc_vs_sc),
    }
    for name, (delta, lo, hi, p) in contrasts.items():
        print(f"{name:28s} Δ={delta:+.4f} CI[{lo:+.4f},{hi:+.4f}] p={p:.4f}")
    if args.out:
        json.dump({"rows": [list(r) for r in rows],
                   "macro": {k: float(np.mean(v)) for k, v in macro.items()},
                   "contrasts": contrasts},
                  open(args.out, "w"), indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
