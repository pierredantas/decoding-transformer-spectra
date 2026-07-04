"""Methodology diagram for the calibration-verified RMT pipeline.
Renders fig_pipeline.pdf (a flowchart) consistent with the paper's style."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.environ.get("FIG_OUT", ".")
plt.rcParams.update({"font.family": "serif", "font.size": 8})

MAIN, MAINE = "#e8eef7", "#1f77b4"     # main-flow boxes
SIG, SIGE = "#fbe4e4", "#d62728"       # signal / spikes
NOISE, NOISEE = "#eeeeee", "#666666"   # noise / bulk
CAL, CALE = "#e5f3e5", "#2ca02c"       # calibration branch

fig, ax = plt.subplots(figsize=(9.2, 3.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def arrow(x1, y1, x2, y2, ec="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.2, color=ec))

# --- main pipeline row ---
W, H, y = 17.0, 12, 27
xs = [1.5, 20.6, 39.7, 58.8, 77.9]
labels = [
    "Pretrained weight\nmatrix $W\\in\\mathbb{R}^{m\\times n}$\n(BERT / ALBERT)",
    "Column-standardize\n+ thin SVD\n$\\lambda_i = s_i^{2}/N$",
    "MP reference\n$\\beta=\\min/\\max$\n+ TW / hard-edge trim",
    "Bulk–spike split\nat the MP edge $\\lambda_+$",
    "KS statistic $D_{\\mathrm{TW}}$\n$\\rightarrow$ bootstrap\naccept / reject",
]
for x, lab in zip(xs, labels):
    box(x, y, W, H, lab, MAIN, MAINE)
for i in range(4):
    arrow(xs[i] + W, y + H / 2, xs[i + 1], y + H / 2)

# --- signal / noise outputs (from the bulk–spike split, box index 3) ---
sx = xs[3] + W / 2
box(38.5, 6, 16, 8, "spikes\n= signal", SIG, SIGE, fs=7)
box(56.5, 6, 18, 8, "bulk $\\approx$ MP\n= noise", NOISE, NOISEE, fs=7)
arrow(sx, y, 46.5, 14, ec=SIGE)
arrow(sx, y, 65.5, 14, ec=NOISEE)

# --- calibration branch feeding the KS decision (box index 4) ---
kx = xs[4] + W / 2
box(77.5, 5, 21, 9,
    "Shape-matched Gaussian\nnull (same $p,N$)\n$\\rightarrow$ bootstrap $p$ + FDR", CAL, CALE, fs=6.8)
arrow(kx, 14, kx, y, ec=CALE)
ax.text(kx + 1.5, 20.5, "calibrate", fontsize=6.5, color=CALE, style="italic", ha="left")

ax.text(50, 42, "Calibration-verified RMT spectral diagnostic",
        ha="center", fontsize=9.5, weight="bold")
fig.tight_layout()
p = os.path.join(OUT, "fig_pipeline.pdf")
fig.savefig(p, bbox_inches="tight"); print("wrote", p)
