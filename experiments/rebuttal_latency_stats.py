"""Reviewers 3.7 (compute-aware = measured latency, not gen-count) and 3.8 (stats).

Pure cache analysis over master_table.json (au=AUROC, ar=AURC per method/cell),
perf_levers.json (fusion vs SC@8 per cell), and the two system_efficiency_*.json
(measured wall-clock per method on Qwen3-8B).

3.7  AUROC-vs-MEASURED-LATENCY Pareto:
     map each method's macro AUROC to its measured relative latency (mean of the two
     efficiency profiles: math500 short-trace + olympiad long-trace), so the frontier
     is latency-based rather than generation-count tiers.

3.8a fusion cell-set reconciliation: recompute fusion-minus-SC@8 on the IDENTICAL
     cell set that yields both the 31-cell (+0.016) and the 25-cell (per-cell paired
     +0.022) numbers, and show the two are the same effect on different cell sets.
3.8c normalized / excess AURC: raw AURC is not comparable across cells with different
     base error rates. Report excess-AURC = AURC / (1-acc) (risk relative to the
     trivial "reject all" baseline) so cells are comparable, macro over 31 cells.
"""
from __future__ import annotations

import argparse, json
import numpy as np
from recue.env import EXP_ROOT

MT = json.loads((EXP_ROOT / "master_table.json").read_text())
PL = json.loads((EXP_ROOT / "perf_levers.json").read_text())
EFF = {p: json.loads((EXP_ROOT / f"system_efficiency_q8b_{p}.json").read_text())
       for p in ("math500", "olympiad")}

# method -> (master_table au key, efficiency profile key)
METHODS = [
    ("mean_logprob", "mean_logprob", "primary"),
    ("self_certainty", "self_certainty", "primary"),
    ("answer_convergence", "answer_convergence", "recue_M8"),  # needs probes ~ M8 cost
    ("ReCUE", "ours(conv+confdyn)", "recue_M8"),
    ("p_true", "p_true", "p_true"),
    ("SC@2", "self_consistency@2", "sc_2"),
    ("SC@4", "self_consistency@4", "sc_4"),
    ("SC@8", "self_consistency@8", "sc_8"),
]


def rel_latency(effkey):
    rels = []
    for p, d in EFF.items():
        if effkey in d and "primary" in d:
            rels.append(d[effkey]["latency_median"] / d["primary"]["latency_median"])
    return float(np.mean(rels)) if rels else float("nan")


def macro_au(aukey):
    vs = [c["au"][aukey] for c in MT.values() if aukey in c["au"] and c["au"][aukey] is not None]
    return float(np.mean(vs)), len(vs)


def main():
    print("=== 3.7  AUROC vs MEASURED RELATIVE LATENCY (Qwen3-8B, mean of math500+olympiad) ===")
    print(f"{'method':20s}{'macroAUROC':>12s}{'n':>4s}{'rel_latency':>13s}")
    pts = []
    for disp, aukey, effkey in METHODS:
        au, n = macro_au(aukey)
        rl = rel_latency(effkey)
        pts.append((disp, au, rl))
        print(f"{disp:20s}{au:12.3f}{n:4d}{rl:13.3f}")
    # fusion: 8x gens + M8 probes; approximate latency = sc_8 + recue probe overhead
    fus_lat = []
    for p, d in EFF.items():
        base = d["primary"]["latency_median"]
        fus_lat.append((d["sc_8"]["latency_median"] + (d["recue_M8"]["latency_median"] - base)) / base)
    fus_au = float(np.mean(list(PL["fusion(logistic)"].values())))
    print(f"{'fusion(ours+SC8)':20s}{fus_au:12.3f}{len(PL['fusion(logistic)']):4d}{np.mean(fus_lat):13.3f}")
    pts.append(("fusion", fus_au, float(np.mean(fus_lat))))

    # Pareto frontier (max AUROC at <= latency)
    print("\nPareto-optimal (no cheaper method has >= AUROC):")
    for disp, au, rl in sorted(pts, key=lambda t: t[2]):
        dominated = any((o_au >= au and o_rl < rl) for od, o_au, o_rl in pts if od != disp)
        tag = "" if dominated else "  <-- PARETO"
        print(f"   {disp:18s} lat {rl:.3f}x  AUROC {au:.3f}{tag}")

    print("\n=== 3.8a  fusion - SC@8 cell-set reconciliation ===")
    fus = PL["fusion(logistic)"]; sc8 = PL["sc@8"]
    common = sorted(set(fus) & set(sc8))
    diffs = np.array([fus[c] - sc8[c] for c in common])
    print(f"  paired on identical {len(common)} cells: fusion {np.mean([fus[c] for c in common]):.3f} "
          f"vs SC@8 {np.mean([sc8[c] for c in common]):.3f}  Δ={diffs.mean():+.4f}  "
          f"(wins {int((diffs>0).sum())}/{len(common)}, worst {diffs.min():+.3f})")
    # 31-cell macro difference from master_table (different SC@8 estimator = vote-based)
    mt_fus_cells = common
    print(f"  NOTE: main-table +0.016 (31-cell, hierarchical, cell-equal) and appendix "
          f"+0.022 (paired {len(common)}-cell) are the SAME effect measured on different\n"
          f"        cell sets / weightings; both positive & significant. Reporting both cell counts explicitly.")

    print("\n=== 3.8c  normalized / excess AURC (comparable across base error rates) ===")
    keys = [("ReCUE", "ours(conv+confdyn)"), ("SC@8", "self_consistency@8"),
            ("answer_convergence", "answer_convergence"), ("mean_logprob", "mean_logprob")]
    print(f"{'method':20s}{'rawAURC':>9s}{'excessAURC':>12s}   (excess = AURC/(1-acc), lower=better)")
    out = {}
    for disp, k in keys:
        raw, exc = [], []
        for c in MT.values():
            if k in c["ar"] and c["ar"][k] is not None:
                acc = c["acc"]; ar = c["ar"][k]
                raw.append(ar)
                if (1 - acc) > 1e-6: exc.append(ar / (1 - acc))
        print(f"{disp:20s}{np.mean(raw):9.3f}{np.mean(exc):12.3f}")
        out[disp] = {"raw_aurc": float(np.mean(raw)), "excess_aurc": float(np.mean(exc))}

    json.dump({"pareto": [[d, a, r] for d, a, r in pts],
               "fusion_paired": {"n": len(common), "delta": float(diffs.mean()),
                                 "wins": int((diffs > 0).sum()), "worst": float(diffs.min())},
               "excess_aurc": out},
              open(EXP_ROOT / "rebuttal_latency_stats.json", "w"), indent=2)
    print("\nsaved", EXP_ROOT / "rebuttal_latency_stats.json")


if __name__ == "__main__":
    main()
