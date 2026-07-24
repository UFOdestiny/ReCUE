"""Risk-coverage curves for 3 representative cells from ladder_feats_pivot.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ["EXP_ROOT"]
d = json.load(open(f"{R}/ladder_feats_pivot.json"))
cells = [("gsm8k_qwen8b_k8", "GSM8K (Qwen3-8B)"),
         ("math500_qwen8b_k8", "MATH500 (Qwen3-8B)"),
         ("olympiad_qwen14b_k8", "OlympiadBench (Qwen3-14B)")]
STYLE = {"passive": ("— TUP", "#888888", "-"),
         "active": ("ARC", "#2c7fb8", "--"),
         "full": ("ReCUE", "#d95f02", "-"),
         "sc8": ("SC@8", "#1b9e77", ":")}
LABEL = {"passive": "TUP (passive)", "active": "ARC (active)", "full": "ReCUE", "sc8": "SC@8 (8$\\times$)"}

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
for ax, (tag, title) in zip(axes, cells):
    rc = d["rc_curves"][tag]
    cov = np.array(rc["coverage"])
    for row in ["passive", "active", "full", "sc8"]:
        _, color, ls = STYLE[row]
        ax.plot(cov, rc["risk"][row], ls, color=color, lw=1.8, label=LABEL[row])
    ax.set_title(f"{title}\nacc={rc['acc']:.2f}", fontsize=9)
    ax.set_xlabel("Coverage", fontsize=9)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Selective risk (error)", fontsize=9)
axes[0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)
plt.tight_layout()
out = "latex/figures/risk_coverage.pdf"
plt.savefig(out, bbox_inches="tight")
print("saved", out)
