"""
RMT spectral-analysis core library.

Reimplementation of the pipeline described in "Decoding Transformers Spectra".
Design decisions here resolve two under-specified points in the manuscript:

(1) SCALE NORMALIZATION (paper issue #1).
    The MP support (1 +/- sqrt(beta))^2 applies to eigenvalues of the sample
    COVARIANCE matrix S = (1/N) A A^T where A has unit-variance entries.
    We therefore compute lambda_i = s_i^2 / N, with s_i the singular values of
    the standardized matrix and N = number of samples (the larger dimension).
    Comparing raw s_i^2 against (1 +/- sqrt(beta))^2 would be off by a factor ~N.

(2) CONSISTENT ORIENTATION (paper issue: beta given as m/n in some rows,
    n/m in others). We always set:
        p (variables, = #eigenvalues) = min(m, n)
        N (samples)                   = max(m, n)
        beta = p / N  in (0, 1]
    and standardize each variable across the N samples.
"""
import numpy as np
from scipy.stats import kstest
# Tracy-Widom (order 1, GOE) mean/std for edge scaling. TW1 std ~ 1.2680.
TW1_STD = 1.2680


def standardize_and_spectrum(A, eps=1e-12):
    """Return (eigenvalues lambda sorted asc, beta, p, N) for matrix A (m x n).

    Orientation: variables = smaller dim, samples = larger dim.
    Each variable is standardized (zero mean, unit variance) across samples.
    lambda_i = singular_value_i(standardized)^2 / N  -> MP scale.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    # orient so rows = variables (p), cols = samples (N), p <= N
    if m <= n:
        X = A                      # p=m variables, N=n samples
    else:
        X = A.T
    p, N = X.shape                 # p <= N guaranteed
    # standardize each variable (row) across its N samples
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=0, keepdims=True)
    sd = np.maximum(sd, eps)
    Xs = (X - mu) / sd
    # singular values of Xs; eigenvalues of (1/N) Xs Xs^T
    s = np.linalg.svd(Xs, compute_uv=False)
    lam = np.sort((s ** 2) / N)
    beta = p / N
    return lam, beta, p, N


def mp_support(beta):
    """MP bulk edges (lambda_minus, lambda_plus) for aspect ratio beta<=1."""
    r = np.sqrt(beta)
    return (1.0 - r) ** 2, (1.0 + r) ** 2


def mp_cdf(x, beta, grid=20000):
    """MP CDF evaluated at array x, via fine numerical integration of density."""
    lm, lp = mp_support(beta)
    x = np.asarray(x, dtype=np.float64)
    t = np.linspace(lm, lp, grid)
    tt = 0.5 * (t[1:] + t[:-1])
    dens = np.sqrt(np.clip((lp - tt) * (tt - lm), 0, None)) / (2 * np.pi * beta * tt)
    cdf_grid = np.concatenate([[0.0], np.cumsum(dens) * (t[1] - t[0])])
    cdf_grid /= cdf_grid[-1]                       # normalize to 1
    out = np.interp(x, t, cdf_grid, left=0.0, right=1.0)
    return out


def tw_margin(beta, N, c_alpha):
    """Adaptive TW trimming half-width: delta = c_alpha * sigma_TW * N^{-2/3}."""
    return c_alpha * TW1_STD * (N ** (-2.0 / 3.0))


def trim_interval(beta, N, c_alpha, eps=1e-9):
    """Return [L, U] interior interval. Square (beta==1) => hard edge at 0."""
    lm, lp = mp_support(beta)
    delta = tw_margin(beta, N, c_alpha)
    if np.isclose(beta, 1.0):
        L = eps                    # exclude hard-edge singularity at 0
        U = lp - delta
    else:
        L = lm + delta
        U = lp - delta
    return L, U


def ks_against_mp(lam, beta, N, c_alpha=None):
    """KS statistic D_p of eigenvalues vs conditional MP.

    c_alpha=None -> strict: FULL empirical spectrum (incl. out-of-support
                    outliers) vs untrimmed MP CDF. sup|ECDF_all - F_MP|.
    c_alpha in {1,2,3} -> KS-TW: restrict to interior [L,U], condition MP CDF.
    Returns (D_p, n_used).
    """
    lm, lp = mp_support(beta)
    if c_alpha is None:
        # strict: keep ALL eigenvalues; MP CDF saturates to 0 below lm, 1 above lp
        if lam.size < 2:
            return np.nan, lam.size
        D = kstest(lam, lambda x: mp_cdf(x, beta)).statistic
        return float(D), int(lam.size)
    # KS-TW: interior interval + conditional MP CDF
    L, U = trim_interval(beta, N, c_alpha)
    sel = lam[(lam >= L) & (lam <= U)]
    if sel.size < 2:
        return np.nan, sel.size
    Fl, Fu = mp_cdf(np.array([L, U]), beta)
    denom = max(Fu - Fl, 1e-12)
    cdf = lambda x: (mp_cdf(x, beta) - Fl) / denom     # conditional MP CDF on [L,U]
    D = kstest(sel, cdf).statistic
    return float(D), int(sel.size)


def bootstrap_pvalue(lam, beta, p, N, c_alpha=None, n_boot=500, seed=0):
    """Bootstrap-calibrated p-value: fraction of synthetic MP-null D >= observed D.

    Null = iid Gaussian matrix of the SAME (p,N) shape, same pipeline.
    Correct calibration for correlated eigenvalues (paper issue #2).
    """
    D_obs, n_used = ks_against_mp(lam, beta, N, c_alpha)
    if not np.isfinite(D_obs):
        return np.nan, D_obs
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_boot):
        G = rng.standard_normal((p, N))
        lam0, b0, _, N0 = standardize_and_spectrum(G)
        D0, _ = ks_against_mp(lam0, b0, N0, c_alpha)
        if np.isfinite(D0) and D0 >= D_obs:
            count += 1
    return (count + 1) / (n_boot + 1), D_obs
