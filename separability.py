"""Formal test that aspect ratio does not order the FFN vs attention families.

Reads out/characterize.csv and, per model, compares the trimmed bulk statistic
D_TW of the beta=0.25 (feed-forward) matrices against the beta=1 (attention)
matrices with a rank-based Mann-Whitney U test (AUC = P(attn > ffn); 0.5 =
indistinguishable), and contrasts the within-attention spread (QKV vs output)
against the between-family gap. Writes out/separability.csv.
"""
import csv, numpy as np
from collections import defaultdict
from scipy.stats import mannwhitneyu

def boot_ci(v, n=5000, seed=1):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    m = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def main():
    rows = [r for r in csv.DictReader(open("out/characterize.csv"))
            if r["degenerate"] == "False"]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["model"]][r["family"]].append(float(r["DTW2"]))
    out = []
    for model in ["bert-base-uncased", "bert-large-uncased"]:
        f = by[model]
        ffn = f["ffn_intermediate"] + f["ffn_output"]      # beta = 0.25
        att = f["attn_qkv"] + f["attn_out"]                # beta = 1
        if not ffn or not att:
            continue
        U, p = mannwhitneyu(att, ffn, alternative="two-sided")
        auc = U / (len(att) * len(ffn))
        gap = abs(np.mean(att) - np.mean(ffn))
        within = abs(np.mean(f["attn_qkv"]) - np.mean(f["attn_out"]))
        rec = dict(model=model, n_ffn=len(ffn), n_att=len(att),
                   ffn_mean=round(boot_ci(ffn)[0], 4), att_mean=round(boot_ci(att)[0], 4),
                   attn_qkv_mean=round(np.mean(f["attn_qkv"]), 4),
                   attn_out_mean=round(np.mean(f["attn_out"]), 4),
                   U=int(U), p=round(float(p), 4), AUC=round(auc, 4),
                   rank_biserial=round(2 * auc - 1, 4),
                   between_gap=round(gap, 4), within_attn_spread=round(within, 4),
                   within_over_between=round(within / gap, 2))
        out.append(rec)
        print(f"{model}: AUC={auc:.3f} p={p:.3f} | between={gap:.3f} "
              f"within={within:.3f} ({within/gap:.1f}x)")
    with open("out/separability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader()
        for r in out: w.writerow(r)
    print("wrote out/separability.csv")

if __name__ == "__main__":
    main()
