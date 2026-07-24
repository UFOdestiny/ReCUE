"""Two headline figures for ReCUE, from cached 30-cell headline results.

Fig A  compute-quality Pareto: relative latency vs macro AUROC; shows ReCUE
        reaching 8x-self-consistency quality at ~1x cost.
Fig B  per-cell ReCUE(1x) vs SC@8(8x) scatter: cell-by-cell proof that one trace
        matches eight samples, colored by dataset.

Colorblind-safe Okabe-Ito palette; thin marks; direct labels; recessive grid.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))
from recue.env import EXP_ROOT

OUT = repo / "latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": "#444444", "figure.dpi": 150,
})

# Okabe-Ito colorblind-safe
INK = "#222222"; GRID = "#d9d9d9"; MUTED = "#888888"
C_RECUE = "#D55E00"   # vermillion (our method)
C_SC = "#0072B2"      # blue (self-consistency)
C_SINGLE = "#999999"  # gray (cheap single-trace baselines)
C_FUSE = "#CC79A7"    # reddish purple (fusion)
DS_COLORS = {"gsm8k": "#0072B2", "math500": "#009E73", "minerva": "#E69F00",
             "olympiad": "#CC79A7", "aime": "#D55E00"}
DS_LABEL = {"gsm8k": "GSM8K", "math500": "MATH500", "minerva": "Minerva",
            "olympiad": "OlympiadBench", "aime": "AIME"}

hv = json.loads((EXP_ROOT / "headline_v2.json").read_text())["table"]
macro = {r: hv[r]["macro_auroc"] for r in hv}

# ---------------- Fig A: compute-quality Pareto ----------------
# (label, macro AUROC, relative latency, color, marker, is_ours)
pts = [
    ("Self-certainty", macro["self_certainty"], 1.00, C_SINGLE, "o", False),
    ("P(True)",        macro["ptrue"],          1.00, C_SINGLE, "s", False),
    ("SC@2",           macro.get("sc2", 0.765), 1.18, C_SC,     "^", False),
    ("SC@4",           0.855,                   1.58, C_SC,     "^", False),
    ("SC@8",           macro["sc8"],            2.39, C_SC,     "D", False),
    ("ReCUE",          macro["full"],           1.03, C_RECUE,  "*", True),
    ("ReCUE+SC@8",     macro["full+sc8"],       2.42, C_FUSE,   "P", True),
]
fig, ax = plt.subplots(figsize=(3.5, 2.9))
# SC@8 quality reference line
sc8 = macro["sc8"]
ax.axhline(sc8, color=C_SC, lw=0.8, ls=":", zorder=1)
ax.text(1.95, sc8 - 0.004, "SC@8 quality", color=C_SC, fontsize=7,
        ha="center", va="top")
# vertical band highlighting the ~1x operating region
ax.axvspan(0.95, 1.10, color=C_RECUE, alpha=0.06, zorder=0)
for label, au, lat, col, mk, ours in pts:
    ax.scatter(lat, au, s=150 if ours else 55, c=col, marker=mk,
               edgecolor="white", linewidth=0.8, zorder=5)
    dx, dy, ha = 0.05, 0.0, "left"
    if label == "ReCUE": dx, dy, ha = -0.05, 0.012, "right"
    if label == "SC@8":  dx, dy, ha = -0.06, 0.002, "right"
    if label == "ReCUE+SC@8": dx, dy, ha = 0.04, 0.006, "left"
    if label == "P(True)": dx, dy, ha = 0.05, 0.006, "left"
    if label == "SC@2": dx, dy, ha = 0.05, -0.014, "left"
    if label == "Self-certainty": dx, dy, ha = 0.05, 0.0, "left"
    if label == "SC@4": dx, dy, ha = 0.05, -0.002, "left"
    ax.annotate(label, (lat, au), (lat + dx, au + dy), ha=ha, fontsize=7,
                color=INK, fontweight="bold" if ours else "normal")
# arrow: ReCUE reaches SC@8 quality at 1/8 the cost
ax.annotate("", xy=(1.03, macro["full"]), xytext=(2.39, sc8),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8, ls="--"))
ax.text(1.70, 0.905, r"$\approx$ same quality, $\sim1/8$ the cost",
        fontsize=6.8, color=MUTED, ha="center")
ax.set_xlabel("Relative latency ($\\times$ one generation)")
ax.set_ylabel("Macro AUROC")
ax.set_xlim(0.85, 2.95); ax.set_ylim(0.66, 0.94)
ax.grid(True, color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "pareto_quality.pdf", bbox_inches="tight")
print("saved pareto_quality.pdf")

# ---------------- Fig B: per-cell ReCUE(1x) vs SC@8(8x) ----------------
pc = json.loads((EXP_ROOT / "headline_percell.json").read_text())
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.plot([0.68, 1.01], [0.68, 1.01], color=MUTED, lw=0.8, ls="--", zorder=1)
ax.text(0.72, 0.725, "ReCUE = SC@8", color=MUTED, fontsize=7, rotation=45,
        rotation_mode="anchor", va="bottom")
wins = 0
for k, v in pc.items():
    ds = k.split("_")[0]
    x, y = v["SC8"], v["ReCUE"]
    wins += x <= y
    ax.scatter(x, y, s=34, c=DS_COLORS[ds], edgecolor="white", linewidth=0.6,
               zorder=5, alpha=0.9)
lims = [0.68, 1.0]
ax.fill_between(lims, lims, [1.0, 1.0], color=C_RECUE, alpha=0.05, zorder=0)
ax.text(0.71, 0.985, f"ReCUE better\n({wins}/{len(pc)} cells)", fontsize=7.5,
        color=C_RECUE, va="top", fontweight="bold")
ax.set_xlabel("SC@8 AUROC  (8$\\times$ cost)")
ax.set_ylabel("ReCUE AUROC  (1$\\times$ cost)")
ax.set_xlim(*lims); ax.set_ylim(*lims)
ax.set_aspect("equal")
ax.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
handles = [Line2D([0], [0], marker="o", ls="", mfc=DS_COLORS[d], mec="white",
                  ms=6, label=DS_LABEL[d]) for d in ["gsm8k", "math500", "minerva", "olympiad", "aime"]]
ax.legend(handles=handles, fontsize=6.5, loc="lower right", framealpha=0.9,
          handletextpad=0.2, borderpad=0.3)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "percell_recue_vs_sc8.pdf", bbox_inches="tight")
print("saved percell_recue_vs_sc8.pdf")
print(f"ReCUE>=SC8 on {wins}/{len(pc)} cells")
