"""Graphical abstract for 'Spectral Conformity Is Not Compressibility'.
Left: the calibrated weight spectrum (MP bulk = noise, spikes = signal).
Center: the title's claim, a bold  != .
Right: the real compression stress-test (truncating weights wrecks the model),
so spectral conformity does NOT imply compressibility."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ---------- palette ----------
BULK, BULK_EDGE = "#c7ccd1", "#9aa1a8"
SIGNAL = "#e8703a"          # orange accent = signal / ASVD
BLUE = "#3b6fb0"            # uniform
GREEN = "#2f9e6f"           # baseline
INK, MUTE = "#2b2b2b", "#8a9098"

fig = plt.figure(figsize=(12.2, 5.3), dpi=200); fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.42,
                       left=0.055, right=0.965, top=0.72, bottom=0.13)

# =========================================================== LEFT: spectrum
axL = fig.add_subplot(gs[0]); axL.set_facecolor("white")
BETA = 0.35
lm, lp = (1-np.sqrt(BETA))**2, (1+np.sqrt(BETA))**2
x = np.linspace(lm, lp, 800)
rho = np.sqrt(np.clip((lp-x)*(x-lm), 0, None))/(2*np.pi*BETA*x)
axL.fill_between(x, rho, color=BULK, zorder=2)
axL.plot(x, rho, color=BULK_EDGE, lw=1.3, zorder=3)
axL.axvline(lp, ls=(0, (4, 3)), color=BULK_EDGE, lw=1.1, zorder=1)
spk, hts = [lp+0.5, lp+0.9, lp+1.5], [0.6, 0.4, 0.27]
for sx, sh in zip(spk, hts):
    axL.plot([sx, sx], [0, sh], color=SIGNAL, lw=3, solid_capstyle="round", zorder=4)
    axL.scatter([sx], [sh], s=34, color=SIGNAL, zorder=5, edgecolor="white", lw=1)
ymax = rho.max()*1.3
axL.set_xlim(lm-0.3, spk[-1]+0.5); axL.set_ylim(0, ymax)
axL.annotate("bulk = noise", (x[np.argmax(rho)], rho.max()*0.5),
             (x[np.argmax(rho)]+0.5, ymax*0.82), fontsize=11.5, color=INK, ha="left",
             arrowprops=dict(arrowstyle="-", color=INK, lw=1, connectionstyle="arc3,rad=-0.2"))
axL.annotate("spikes = signal", (spk[0], hts[0]), (spk[0]-0.15, ymax*0.95),
             fontsize=11.5, color=SIGNAL, ha="left")
axL.text(lp+0.12, ymax*0.12, "MP edge", fontsize=9.5, color=BULK_EDGE, style="italic", ha="left")
axL.set_xlabel("Eigenvalue", fontsize=11.5, color=INK)
axL.set_ylabel("Spectral density", fontsize=11.5, color=INK)
axL.set_xticks([]); axL.set_yticks([])
for s in ("top", "right"): axL.spines[s].set_visible(False)
for s in ("left", "bottom"): axL.spines[s].set_color(INK)
# calibration badge
axL.text(0.5, -0.235, "Calibration-verified null:  D ≈ 0.005  for any shape",
         transform=axL.transAxes, fontsize=10, color=GREEN, ha="center", va="top",
         bbox=dict(boxstyle="round,pad=0.45", fc="#eaf6f0", ec=GREEN, lw=1))
axL.text(0.5, 1.06, "SPECTRAL CONFORMITY", transform=axL.transAxes, ha="center",
         fontsize=12.5, color=INK, fontweight="bold")

# =========================================================== RIGHT: compression
axR = fig.add_subplot(gs[1]); axR.set_facecolor("white")
kept = [0.7, 0.5, 0.3]                       # real BERT-base means (results/compression_multi.csv)
uniform, asvd, base = [4.80, 6.56, 7.71], [2.62, 3.55, 6.51], 2.21
axR.axhline(base, ls="--", color=GREEN, lw=1.4, zorder=1)
axR.text(0.305, base+0.15, "uncompressed (2.21)", fontsize=9.5, color=GREEN, ha="left")
axR.plot(kept, uniform, "o-", color=BLUE, lw=2.2, ms=6, label="plain SVD (uniform)")
axR.plot(kept, asvd, "s-", color=SIGNAL, lw=2.2, ms=6, label="activation-aware (ASVD)")
axR.fill_between(kept, asvd, base, color="#f6d9c9", alpha=0.5, zorder=0)
axR.annotate("no free\ncompression", (0.5, (asvd[1]+base)/2), fontsize=10.5,
             color=INK, ha="center", va="center", fontweight="medium")
axR.set_xlim(0.75, 0.27); axR.set_ylim(1.6, 8.4)     # invert x: more compression -> right
axR.set_xlabel("fraction of parameters kept", fontsize=11.5, color=INK)
axR.set_ylabel("masked-LM loss  (↓ better)", fontsize=11.5, color=INK)
axR.tick_params(labelsize=9.5, colors=INK)
for s in ("top", "right"): axR.spines[s].set_visible(False)
for s in ("left", "bottom"): axR.spines[s].set_color(INK)
axR.legend(fontsize=9, loc="upper left", frameon=False)
axR.text(0.5, 1.06, "COMPRESSIBILITY", transform=axR.transAxes, ha="center",
         fontsize=12.5, color=INK, fontweight="bold")

# =========================================================== center  !=
fig.text(0.503, 0.42, "≠", ha="center", va="center", fontsize=54,
         color=SIGNAL, fontweight="bold")

# =========================================================== title header
fig.text(0.5, 0.95, "Spectral Conformity Is Not Compressibility",
         ha="center", fontsize=18, color=INK, fontweight="bold")
fig.text(0.5, 0.88, "A Calibration-Verified Random Matrix Study of Transformer Weights",
         ha="center", fontsize=12.5, color=MUTE, style="italic")

for ext in ("pdf", "png"):
    fig.savefig(f"graphical_abstract.{ext}", bbox_inches="tight", facecolor="white")
print("wrote graphical_abstract.pdf / .png")
