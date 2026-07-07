"""B2 (review Weakness 4 / Question 3): ground the bulk-spike split in spiked-model
theory and give it a BBP-informed spike cut.

The paper separates "spikes" (signal) from "bulk" (noise) at the bare MP upper edge
lambda_+ = (1+sqrt(beta))^2 (spikes.py). The referee's concern: that boundary is
exactly the Baik-Ben Arous-Peche (BBP) DETECTABILITY threshold, so a bare-edge cut
sits on the phase-transition line where a sample eigenvalue just above lambda_+ is
ambiguous -- it may be a genuine near-critical spike OR a Tracy-Widom fluctuation of
the bulk maximum (of order N^{-2/3}). A bare-lambda_+ cut therefore OVER-counts
spikes, and those spike counts drive the spectral compression allocation.

This module supplies:
  * the spiked-model theory (BBP threshold; the Benaych-Georges-Nadakuditi outlier
    map and its inverse), so outliers can be reported as implied population spikes;
  * a generalised split with a TW-margined ("BBP-informed") cut lambda_+ + delta,
    which only calls an eigenvalue a spike if it clears the bulk edge by more than
    the expected TW fluctuation.

Theory references to ADD to the manuscript (Sec. 2.2 / 4.3):
  - Baik, Ben Arous, Peche (2005), Ann. Probab. -- the BBP phase transition.
  - Benaych-Georges & Nadakuditi (2011), "The eigenvalues and eigenvectors of finite,
    low-rank perturbations of large random matrices" -- the outlier location map.
  (Verify exact venue/year before citing; see scaffolds/README.md caveat.)

Conventions match rmt.py: variables p=min(m,n), samples N=max(m,n), beta=p/N in (0,1],
bulk variance normalised to 1, so the MP upper edge is lambda_+ = (1+sqrt(beta))^2.
"""
import numpy as np
from rmt import mp_support, tw_margin


# ---------- spiked-model theory (population spike theta, bulk variance = 1) ----------

def bbp_threshold(beta):
    """Population-spike detectability threshold: theta must exceed 1+sqrt(beta) to
    produce a sample-eigenvalue outlier separable from the MP bulk edge."""
    return 1.0 + np.sqrt(beta)


def outlier_location(theta, beta):
    """Benaych-Georges-Nadakuditi: asymptotic sample outlier for a population spike
    theta. For theta <= 1+sqrt(beta) the outlier is absorbed into the edge lambda_+."""
    theta = np.asarray(theta, float)
    _, lp = mp_support(beta)
    lam = theta + beta * theta / (theta - 1.0)
    return np.where(theta > bbp_threshold(beta), lam, lp)


def implied_population_spike(lam_obs, beta):
    """Inverse map: given an observed outlier lambda, recover the implied population
    spike theta (larger root of theta^2 + (beta-1-lambda) theta + lambda = 0).
    Returns NaN when lambda <= lambda_+ (no separable population spike)."""
    lam = np.asarray(lam_obs, float)
    _, lp = mp_support(beta)
    b = (beta - 1.0 - lam)
    disc = b * b - 4.0 * lam
    disc = np.clip(disc, 0.0, None)
    theta = (-b + np.sqrt(disc)) / 2.0
    return np.where(lam > lp, theta, np.nan)


# ---------- generalised signal/noise split with a selectable cut ----------

def spike_cut(beta, N, c_alpha=None):
    """Threshold above which an eigenvalue is called a spike.
      c_alpha is None -> bare MP edge lambda_+  (reproduces spikes.signal_noise_split)
      c_alpha in {1,2,3} -> BBP-informed cut lambda_+ + c_alpha * sigma_TW * N^{-2/3}
    Using the SAME c_alpha as the D_TW trim keeps the spike side and the bulk side
    consistent about what counts as an edge fluctuation."""
    _, lp = mp_support(beta)
    if c_alpha is None:
        return lp
    return lp + tw_margin(beta, N, c_alpha)


def signal_noise_split_bbp(lam, beta, N, c_alpha=None):
    """Drop-in generalisation of spikes.signal_noise_split with a selectable cut.

    Returns the original fields (n_spikes, spike_frac, spike_energy, signal_rank,
    n_below, bulk) plus:
      cut               : the spike threshold actually used
      theta_spikes      : implied population spikes for each retained outlier
      near_edge_spikes  : outliers between lambda_+ and the BBP-informed cut, i.e.
                          the spikes the bare-edge rule keeps but the BBP rule drops
    """
    lam = np.asarray(lam, float)
    lm, lp = mp_support(beta)
    cut = spike_cut(beta, N, c_alpha)
    spikes = lam[lam > cut]
    bulk = lam[(lam >= lm) & (lam <= lp)]
    total = float(lam.sum()) if lam.sum() > 0 else 1.0
    return dict(
        n_spikes=int(spikes.size),
        spike_frac=float(spikes.size) / lam.size,
        spike_energy=float(spikes.sum()) / total,
        signal_rank=int(spikes.size),
        n_below=int((lam < lm).sum()),
        bulk=bulk,
        cut=float(cut),
        theta_spikes=implied_population_spike(spikes, beta),
        near_edge_spikes=int(((lam > lp) & (lam <= cut)).sum()),
    )


if __name__ == "__main__":
    # sanity: at the critical population spike the outlier merges into lambda_+,
    # and inverting lambda_+ recovers exactly the BBP threshold.
    for beta in (1.0, 0.25, 0.03):
        _, lp = mp_support(beta)
        thr = bbp_threshold(beta)
        assert np.isclose(outlier_location(thr, beta), lp), beta
        assert np.isclose(implied_population_spike(lp + 1e-9, beta), thr, atol=1e-3), beta
        print(f"beta={beta:4.2f}  lambda_+={lp:.4f}  BBP theta*={thr:.4f}  "
              f"outlier(1.5*theta*)={float(outlier_location(1.5*thr, beta)):.4f}")
    print("spiked-model identities check out.")
