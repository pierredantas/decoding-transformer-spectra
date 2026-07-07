"""Multi-model compression figure with error bands + Gavish-Donoho baseline.
Reads out/compression_multi.csv (long form) -> fig_real_compression_multi.pdf."""
import csv, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict

OUT = os.environ.get("FIG_OUT", ".")
COL = {"uniform": "#1f77b4", "spectral": "#d62728", "random": "#7f7f7f", "gd": "#2ca02c",
       "asvd": "#9467bd", "fwsvd": "#ff7f0e"}
LAB = {"uniform": "uniform", "spectral": "spectral-guided", "random": "random",
       "gd": "Gavish–Donoho", "asvd": "ASVD", "fwsvd": "FWSVD"}
NAME = {"bert-base-uncased": "BERT-base", "bert-large-uncased": "BERT-large",
        "albert-base-v2": "ALBERT"}

def load(path="out/compression_multi.csv"):
    d = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model->mode->kept->[loss]
    base = {}
    for r in csv.DictReader(open(path)):
        model, mode = r["model"], r["mode"]; kf = float(r["kept_frac"]); L = float(r["mlm_loss"])
        if mode == "baseline": base.setdefault(model, []).append(L)
        else: d[model][mode][kf].append(L)
    return d, {m: np.mean(v) for m, v in base.items()}

def main():
    d, base = load()
    # ALBERT excluded from the compression study: its MLM baseline (~12.3) is broken,
    # so degradation is not measurable. ALBERT remains in the spectral characterization.
    models = [m for m in ["bert-base-uncased", "bert-large-uncased"] if m in d]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), squeeze=False)
    for ax, model in zip(axes[0], models):
        for mode in ["uniform", "spectral", "random", "asvd", "fwsvd"]:
            pts = sorted(d[model][mode].items())
            if not pts: continue
            xs = [k for k, _ in pts]; ys = [np.mean(v) for _, v in pts]; es = [np.std(v) for _, v in pts]
            ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=4, lw=1.3, capsize=2,
                        color=COL[mode], label=LAB[mode])
        if d[model].get("gd"):
            gk = list(d[model]["gd"].keys())[0]; gy = np.mean(d[model]["gd"][gk])
            ax.scatter([gk], [gy], marker="*", s=120, color=COL["gd"],
                       edgecolor="k", zorder=5, label=LAB["gd"])
        if model in base:
            ax.axhline(base[model], ls="--", c="green", lw=1, label=f"baseline ({base[model]:.2f})")
        ax.set_title(NAME.get(model, model), fontsize=9)
        ax.set_xlabel("fraction of SVD parameters kept"); ax.invert_xaxis(); ax.grid(alpha=0.3)
        if ax is axes[0][0]: ax.set_ylabel("masked-LM loss (WikiText-2)")
        ax.legend(fontsize=6)
    fig.suptitle("Compression across models: plain-SVD allocations (uniform / spectral-guided / random) "
                 "vs activation-aware bases (ASVD / FWSVD) vs Gavish–Donoho "
                 "(mean ± s.d. over seeds, no fine-tuning)", fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_real_compression_multi.pdf")
    fig.savefig(p, bbox_inches="tight"); print("wrote", p)

    # also print a compact summary table
    print("\nmodel        kept   uniform  spectral  random")
    for model in models:
        ks = sorted(set(d[model]["uniform"]) & set(d[model]["spectral"]) & set(d[model]["random"]))
        for kf in ks:
            u = np.mean(d[model]["uniform"][kf]); s = np.mean(d[model]["spectral"][kf]); r = np.mean(d[model]["random"][kf])
            print(f"{NAME.get(model,model):11s} {kf:.2f}   {u:6.2f}   {s:6.2f}   {r:6.2f}")

if __name__ == "__main__":
    main()
