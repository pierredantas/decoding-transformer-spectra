"""A1 (review Weakness 1 / Question 1): head-to-head between the paper's
calibrated D_TW and the incumbent WeightWatcher power-law alpha metric.

Thesis under test: "prior spectral tests are miscalibrated". This module makes
that claim *demonstrable* by computing, on the SAME matrices the paper already
analyses, both:
  * D_TW   -- the paper's trimmed, calibration-verified KS statistic (rmt.py)
  * alpha  -- the Martin-Mahoney heavy-tailed power-law exponent (WeightWatcher)

and reporting where a D_TW verdict AGREES with vs. OVERTURNS an alpha-based read.

Drop this file into the repo root (next to rmt.py / extract.py) and run:
    python weightwatcher_compare.py bert-base-uncased bert-large-uncased albert-base-v2
Outputs out/ww_compare.csv and prints Spearman rho(D_TW, alpha) overall + per family.

alpha is SCALE-FREE, so it does not matter that WeightWatcher fits the raw ESD
(eigenvalues s_i^2) while the paper works on lambda_i = s_i^2 / N -- the exponent
is identical under that rescaling. We therefore fit alpha on s_i^2 directly.

The alpha fit below is a self-contained Clauset-Shalizi-Newman (CSN) MLE with
KS-minimising x_min selection -- the same estimator WeightWatcher uses for its
'PL' fit. If the real `weightwatcher` package is installed, pass --use-ww to call
it instead and cross-check (recommended for the final camera-ready numbers).
"""
import sys, csv, os, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from extract import extract
from rmt import standardize_and_spectrum, ks_against_mp

C_ALPHA = 2          # matches D_TW(c=2) reported in the paper (run_characterize.py)
MIN_EIG = 10         # skip degenerate matrices, as the paper does
MIN_TAIL = 10        # need at least this many tail eigenvalues for a stable alpha fit


def powerlaw_alpha(evals):
    """CSN power-law MLE with KS-minimising x_min. Returns (alpha, xmin, D_ks, n_tail).

    Fits p(x) ~ x^{-alpha} to the upper tail of the empirical spectral density.
    `evals` are eigenvalues (s_i^2); zeros/negatives dropped. alpha in ~[2,6] for
    trained nets; smaller alpha = heavier tail = more "structured" per WeightWatcher.
    """
    ev = np.sort(evals[evals > 0])
    if ev.size < MIN_TAIL + 5:
        return np.nan, np.nan, np.nan, 0
    best = (np.inf, np.nan, np.nan, 0)          # (D_ks, alpha, xmin, n)
    # candidate x_min = each eigenvalue that still leaves >= MIN_TAIL in the tail
    for i in range(ev.size - MIN_TAIL):
        xmin = ev[i]
        tail = ev[ev >= xmin]
        n = tail.size
        s = np.sum(np.log(tail / xmin))
        if s <= 0:
            continue
        alpha = 1.0 + n / s
        # KS distance between empirical tail CDF and the fitted power-law CDF
        cdf_emp = np.arange(1, n + 1) / n
        cdf_fit = 1.0 - (tail / xmin) ** (1.0 - alpha)
        D = np.max(np.abs(cdf_emp - cdf_fit))
        if D < best[0]:
            best = (D, alpha, xmin, n)
    D, alpha, xmin, n = best
    return alpha, xmin, D, n


def alpha_via_weightwatcher(W):
    """Optional cross-check against the real `weightwatcher` package (--use-ww).
    Returns alpha or NaN. Kept import-local so the module runs without the dep."""
    try:
        import weightwatcher as ww           # noqa: F401
        import torch, torch.nn as nn
        lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(W).float())
        watcher = ww.WeightWatcher(model=lin)
        d = watcher.analyze(mp_fit=False)
        return float(d["alpha"].iloc[0])
    except Exception as e:                    # dep missing or API drift -> skip
        print(f"    [weightwatcher unavailable: {e}]", flush=True)
        return np.nan


def run(model_id, use_ww=False):
    rows = []
    for m in extract(model_id):
        lam, beta, p, N = standardize_and_spectrum(m["W"])
        if p < MIN_EIG:
            continue
        d_tw, _ = ks_against_mp(lam, beta, N, C_ALPHA)
        # WeightWatcher fits the ESD; lambda_i * N == s_i^2 recovers the raw ESD,
        # but alpha is scale-free so we fit on lam directly.
        alpha, xmin, d_pl, n_tail = powerlaw_alpha(lam)
        if use_ww:
            alpha_ww = alpha_via_weightwatcher(m["W"])
        else:
            alpha_ww = np.nan
        rows.append(dict(model=model_id, name=m["name"], family=m["family"],
                         beta=round(beta, 4), D_TW=round(float(d_tw), 4),
                         alpha=round(float(alpha), 3) if np.isfinite(alpha) else np.nan,
                         alpha_ww=round(alpha_ww, 3) if np.isfinite(alpha_ww) else np.nan,
                         xmin=xmin, pl_ks=round(float(d_pl), 4), n_tail=n_tail))
        print(f"  {m['family']:16s} beta={beta:4.2f} D_TW={d_tw:.3f} "
              f"alpha={alpha:5.2f} (tail n={n_tail})", flush=True)
    return rows


def report(rows):
    dtw = np.array([r["D_TW"] for r in rows], float)
    alp = np.array([r["alpha"] for r in rows], float)
    ok = np.isfinite(dtw) & np.isfinite(alp)
    rho, pv = spearmanr(dtw[ok], alp[ok])
    print(f"\n=== D_TW vs alpha ===  Spearman rho = {rho:+.3f} (p={pv:.1e}, n={ok.sum()})")
    print("Expect rho < 0: higher D_TW (more bulk deviation) should track lower alpha")
    print("(heavier tail). Matrices where the sign DISAGREES are exactly the cases")
    print("where calibration overturns an alpha-based verdict -- report these.\n")
    by = defaultdict(list)
    for r in rows:
        by[r["family"]].append(r)
    print(f"{'family':16s} {'n':>3} {'mean D_TW':>10} {'mean alpha':>11}")
    for fam, rs in sorted(by.items()):
        md = np.nanmean([r["D_TW"] for r in rs])
        ma = np.nanmean([r["alpha"] for r in rs])
        print(f"{fam:16s} {len(rs):3d} {md:10.3f} {ma:11.2f}")


if __name__ == "__main__":
    use_ww = "--use-ww" in sys.argv
    mids = [a for a in sys.argv[1:] if not a.startswith("--")] or \
           ["bert-base-uncased", "bert-large-uncased", "albert-base-v2"]
    allrows = []
    for mid in mids:
        print(f"\n########## {mid} ##########", flush=True)
        allrows += run(mid, use_ww=use_ww)
    report(allrows)
    os.makedirs("out", exist_ok=True)
    with open("out/ww_compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    print("wrote out/ww_compare.csv")
