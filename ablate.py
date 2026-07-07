"""C3 (review Weakness 7 / Question 8): robustness of the qualitative conclusions to
the pipeline's own remaining free parameters.

The paper's thesis is that *implicit* choices flip spectral conclusions; a referee
rightly asks whether the paper's *own* choices do. This driver sweeps them and shows
the two headline conclusions are invariant:
  (H1) family ordering by D_TW           -- rank-stable across settings
  (H2) aspect ratio does not separate    -- attention/FFN gap stays small & unstable
                                            relative to within-beta spread

Knobs swept:
  * trim margin  c_alpha in {1, 2, 3}         (rmt.trim_interval)
  * standardize floor  eps in {1e-12,1e-8,1e-6,1e-4}   (rmt.standardize_and_spectrum)
  * null draws  K in {50,100,200,400}         (Monte-Carlo convergence of the null)
Plus per-family Benjamini-Hochberg FDR q-values at the baseline setting.

Drop into repo root; run (needs transformers to load pinned weights):
    python ablate.py bert-base-uncased bert-large-uncased
Writes out/ablate_grid.csv, out/ablate_fdr.csv, out/ablate_nullconv.csv and prints
three summary tables. Manuscript deliverable: one supplementary table + a sentence
"conclusions invariant across c_alpha, eps (family-order Spearman >= 〔rho〕)".
"""
import sys, csv, os, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, mannwhitneyu
from extract import extract
from rmt import standardize_and_spectrum, ks_against_mp

MIN_EIG = 10
C_ALPHAS = [1, 2, 3]
EPSES = [1e-12, 1e-8, 1e-6, 1e-4]
KS = [50, 100, 200, 400]
BASE = (2, 1e-12)                      # baseline (c_alpha, eps) matching the paper
ATTN = {"attn_qkv", "attn_out"}
FFN = {"ffn_intermediate", "ffn_output"}

_NULL = {}
def null_dtw(p, N, c_alpha, K, seed=12345):
    """Sorted null distribution of D_TW for iid Gaussian of shape (p,N)."""
    key = (p, N, c_alpha, K)
    if key in _NULL:
        return _NULL[key]
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(K):
        lam, b, _, NN = standardize_and_spectrum(rng.standard_normal((p, N)))
        d, _ = ks_against_mp(lam, b, NN, c_alpha)
        if np.isfinite(d):
            ds.append(d)
    val = np.sort(ds)
    _NULL[key] = val
    return val

def pval(null_sorted, obs):
    return float((np.sum(null_sorted >= obs) + 1) / (null_sorted.size + 1))

def bh_qvalues(pvals):
    """Benjamini-Hochberg FDR q-values."""
    p = np.asarray(pvals, float); n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out


def grid(model_id, rows):
    """Per-family mean D_TW over the (c_alpha, eps) grid + H1/H2 invariance."""
    mats = [m for m in extract(model_id)]
    fam_by_setting = {}
    for c_alpha in C_ALPHAS:
        for eps in EPSES:
            fam = defaultdict(list); attn_vals, ffn_vals = [], []
            for m in mats:
                lam, beta, p, N = standardize_and_spectrum(m["W"], eps=eps)
                if p < MIN_EIG:
                    continue
                d, _ = ks_against_mp(lam, beta, N, c_alpha)
                if not np.isfinite(d):
                    continue
                fam[m["family"]].append(d)
                if m["family"] in ATTN: attn_vals.append(d)
                if m["family"] in FFN: ffn_vals.append(d)
            means = {k: float(np.mean(v)) for k, v in fam.items()}
            fam_by_setting[(c_alpha, eps)] = means
            # H2: attention-vs-FFN separability (AUC 0.5 == indistinguishable)
            if attn_vals and ffn_vals:
                U = mannwhitneyu(attn_vals, ffn_vals, alternative="two-sided")
                auc = U.statistic / (len(attn_vals) * len(ffn_vals))
            else:
                auc = np.nan
            rows.append(dict(model=model_id, c_alpha=c_alpha, eps=eps,
                             auc_attn_vs_ffn=round(auc, 3),
                             **{f"D_{k}": round(v, 4) for k, v in means.items()}))
    # H1: rank stability of family ordering vs baseline
    base = fam_by_setting[BASE]
    fams = sorted(base)
    print(f"\n=== {model_id}: family-order stability (Spearman vs baseline c=2,eps=1e-12) ===")
    worst = 1.0
    for setting, means in fam_by_setting.items():
        common = [f for f in fams if f in means]
        rho, _ = spearmanr([base[f] for f in common], [means[f] for f in common])
        worst = min(worst, rho)
    print(f"  worst-case family-order Spearman across all settings: {worst:.3f}  (want ~1.0)")
    aucs = [r["auc_attn_vs_ffn"] for r in rows if r["model"] == model_id]
    print(f"  attn-vs-FFN AUC range across settings: "
          f"[{np.nanmin(aucs):.3f}, {np.nanmax(aucs):.3f}]  (0.5 == no separation)")


def fdr(model_id, rows):
    """Per-family BH q-values at the baseline setting."""
    c_alpha, eps = BASE
    by_fam = defaultdict(list)
    for m in extract(model_id):
        lam, beta, p, N = standardize_and_spectrum(m["W"], eps=eps)
        if p < MIN_EIG:
            continue
        d, _ = ks_against_mp(lam, beta, N, c_alpha)
        pv = pval(null_dtw(p, N, c_alpha, 200), d)
        by_fam[m["family"]].append(pv)
    print(f"\n=== {model_id}: per-family FDR (BH) at baseline ===")
    print(f"{'family':16s} {'n':>3} {'median q':>9} {'max q':>7}")
    for fam, pv in sorted(by_fam.items()):
        q = bh_qvalues(pv)
        rows.append(dict(model=model_id, family=fam, n=len(pv),
                         median_q=round(float(np.median(q)), 5),
                         max_q=round(float(np.max(q)), 5)))
        print(f"{fam:16s} {len(pv):3d} {np.median(q):9.5f} {np.max(q):7.5f}")


def null_convergence(model_id, rows):
    """MC convergence of the null (mean/p95) vs number of draws K, for the shapes present."""
    shapes = sorted({(min(m["shape"]), max(m["shape"])) for m in extract(model_id)
                     if min(m["shape"]) >= MIN_EIG})[:5]
    print(f"\n=== {model_id}: null D_TW convergence (c=2) ===")
    print(f"{'(p,N)':>16} " + " ".join(f"K={k:>4}" for k in KS))
    for (p, N) in shapes:
        line, meanrow = [], {}
        for K in KS:
            nd = null_dtw(p, N, 2, K)
            m = float(nd.mean()); line.append(f"{m:.4f}")
            meanrow[f"mean_K{K}"] = round(m, 4)
        rows.append(dict(model=model_id, p=p, N=N, **meanrow))
        print(f"{str((p,N)):>16} " + " ".join(f"{x:>6}" for x in line))


if __name__ == "__main__":
    mids = sys.argv[1:] or ["bert-base-uncased", "bert-large-uncased"]
    grid_rows, fdr_rows, nc_rows = [], [], []
    for mid in mids:
        grid(mid, grid_rows); fdr(mid, fdr_rows); null_convergence(mid, nc_rows)
    os.makedirs("out", exist_ok=True)
    for fname, rws in [("ablate_grid.csv", grid_rows),
                       ("ablate_fdr.csv", fdr_rows),
                       ("ablate_nullconv.csv", nc_rows)]:
        keys = sorted({k for r in rws for k in r})
        with open(f"out/{fname}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rws)
    print("\nwrote out/ablate_grid.csv, out/ablate_fdr.csv, out/ablate_nullconv.csv")
