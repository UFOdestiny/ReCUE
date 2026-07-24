"""Build once, slice fast. Dump per-cell feature blocks + labels to one npz so the
ladder algebra (and decompositions) run without re-paying math_verify _eq costs.

Blocks saved per cell (id-joined across probe/conf/gen/rebuttal/labels caches):
  y          : labels
  AGREE      : [primary--final agreement]                     (1 col)
  FLL        : [full-answer len-norm loglik, min tok lp, first-2 mean]   (3 cols)
  FCONF      : [final first-answer-token logprob]             (1 col)
  CONVP      : [answer-convergence scalar]                    (1 col, multi-depth-derived)
  MULTI      : DualCommit CONV(6)+CDYN(6)                      (12 cols)
  TRACE      : TraceProfile / UTP                             (27 cols)
"""
from __future__ import annotations
import argparse, json
import numpy as np
from recue.env import EXP_ROOT
from recue.features import _eq

NBIN = 8


def conv_cdyn(neu, lp):
    n = len(neu); final = neu[-1] if neu else None; half = n // 2
    agree = [1.0 if (a is not None and final is not None and _eq(a, final)) else 0.0 for a in neu]
    af = np.mean(agree) if n else 0.0
    lh = np.mean(agree[half:]) if n - half > 0 else af
    run = 0
    for i in range(n - 1, -1, -1):
        if agree[i] == 1.0: run += 1
        else: break
    fst = run / n if n else 0.0
    conv = 1.0
    for i in range(n):
        if all(agree[j] == 1.0 for j in range(i, n)): conv = (i + 1) / n; break
    ids, reps = [], []
    for a in neu:
        if a is None: ids.append(-1); continue
        f = next((k for k, r in enumerate(reps) if _eq(a, r)), None)
        if f is None: reps.append(a); f = len(reps) - 1
        ids.append(f)
    flip = sum(1 for i in range(1, n) if ids[i] != ids[i - 1]) / max(1, n - 1)
    CONV = [af, lh, fst, conv, flip, len(set(i for i in ids if i != -1))]
    lpv = [v for v in lp if v is not None]
    slope = float(np.polyfit(np.arange(len(lpv)), lpv, 1)[0]) if len(lpv) >= 2 else 0.0
    fc = -10.0
    for a, v in zip(neu, lp):
        if a is not None and final is not None and _eq(a, final) and v is not None:
            fc = v; break
    CDYN = [float(np.mean(lpv)) if lpv else -10.0, lpv[-1] if lpv else -10.0,
            float(np.min(lpv)) if lpv else -10.0, slope, fc,
            float(np.std(lpv)) if len(lpv) > 1 else 0.0]
    return CONV + CDYN, conv


def _entropy_traj(topk):
    ents = []
    for vals in topk:
        if not vals: continue
        p = np.exp(np.array(vals)); p = p / max(p.sum(), 1e-12)
        ents.append(-float(np.sum(p * np.log(p + 1e-12))))
    return np.array(ents)


def _binned(arr, nbin):
    if len(arr) == 0: return [0.0] * nbin
    idx = np.linspace(0, len(arr), nbin + 1).astype(int); out = []
    for i in range(nbin):
        seg = arr[idx[i]:idx[i + 1]]
        out.append(float(np.mean(seg)) if len(seg) else float(arr[min(idx[i], len(arr) - 1)]))
    return out


def _slope_r2(arr):
    n = len(arr)
    if n < 3: return 0.0, 0.0
    x = np.arange(n); b1, b0 = np.polyfit(x, arr, 1); pred = b1 * x + b0
    ss_res = np.sum((arr - pred) ** 2); ss_tot = np.sum((arr - np.mean(arr)) ** 2)
    return float(b1), float(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)


def traceprofile(g):
    lp = np.array(g.get("chosen_logprobs") or [], dtype=float)
    ent = _entropy_traj(g.get("topk_logprobs") or [])
    if len(lp) == 0: return [0.0] * (2 * NBIN + 11)
    feats = _binned(lp, NBIN) + (_binned(ent, NBIN) if len(ent) else [0.0] * NBIN)
    feats += [float(np.mean(lp)), float(np.min(lp)), float(np.std(lp))]
    feats += [float(np.mean(ent)) if len(ent) else 0.0, float(np.max(ent)) if len(ent) else 0.0]
    s_lp, r2_lp = _slope_r2(lp); s_e, r2_e = _slope_r2(ent) if len(ent) else (0.0, 0.0)
    feats += [s_lp, r2_lp, s_e, r2_e, float(np.mean(lp[-30:])), float(np.mean(lp < -2.0))]
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    blob = {}
    for tag in args.tags:
        pf = EXP_ROOT / "probe" / f"{tag}_probe.json"
        cf = EXP_ROOT / "conf" / f"{tag}_conf.json"
        lf = EXP_ROOT / "labels" / f"{tag}.json"
        gf = EXP_ROOT / "gen" / f"{tag}.json"
        rf = EXP_ROOT / "rebuttal" / f"{tag}_reb.json"
        ptf = EXP_ROOT / "ptrue" / f"{tag}_ptrue.json"
        if not (pf.exists() and cf.exists() and lf.exists() and gf.exists() and rf.exists()):
            continue
        probe = {r["id"]: r for r in json.loads(pf.read_text())}
        conf = json.loads(cf.read_text()); labs = json.loads(lf.read_text())
        gen = {g["id"]: g for g in json.loads(gf.read_text())}
        reb = {r["id"]: r for r in json.loads(rf.read_text())}
        ptrue = json.loads(ptf.read_text()) if ptf.exists() else {}
        y, AGREE, FLL, FCONF, CONVP, MULTI, TRACE, PT = [], [], [], [], [], [], [], []
        for r in conf:
            rid = r["id"]
            if (not r["intermediate"] or rid not in labs or rid not in probe
                    or rid not in gen or rid not in reb):
                continue
            neu = [x["neutral"] for x in r["intermediate"]]
            lp = [x.get("neutral_lp") for x in r["intermediate"]]
            primary = probe[rid].get("final_answer")
            a_J = neu[-1] if neu else None
            fc_last = lp[-1] if lp and lp[-1] is not None else -10.0
            agree = 1.0 if (primary is not None and a_J is not None and _eq(primary, a_J)) else 0.0
            full = reb[rid].get("full") or {}
            tl = full.get("tok_lps") or []
            fll = float(np.mean(tl)) if tl else -10.0
            fmin = float(np.min(tl)) if tl else -10.0
            f2 = float(np.mean(tl[:2])) if len(tl) >= 2 else fll
            mult, convp = conv_cdyn(neu, lp)
            AGREE.append([agree]); FLL.append([fll, fmin, f2]); FCONF.append([fc_last])
            CONVP.append([convp]); MULTI.append(mult); TRACE.append(traceprofile(gen[rid]))
            PT.append([ptrue.get(rid, np.nan)])
            y.append(labs[rid])
        y = np.array(y)
        if len(y) < 30 or y.sum() == 0 or (1 - y).sum() == 0:
            continue
        blob[f"{tag}::y"] = y
        blob[f"{tag}::AGREE"] = np.array(AGREE)
        blob[f"{tag}::FLL"] = np.array(FLL)
        blob[f"{tag}::FCONF"] = np.array(FCONF)
        blob[f"{tag}::CONVP"] = np.array(CONVP)
        blob[f"{tag}::MULTI"] = np.array(MULTI)
        blob[f"{tag}::TRACE"] = np.array(TRACE)
        blob[f"{tag}::PT"] = np.array(PT)
        print(f"dumped {tag} n={len(y)}")
    np.savez(args.out, **blob)
    print("saved", args.out, "cells:", len({k.split('::')[0] for k in blob}))


if __name__ == "__main__":
    main()
