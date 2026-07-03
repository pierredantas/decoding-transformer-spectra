"""Fast, deterministic invariant tests for the RMT pipeline.

These do NOT download large models; they validate the math and calibration so CI
can guard correctness in seconds. The heavy end-to-end reproduction lives in the
full-reproduction workflow.
"""
import numpy as np
import pytest
from rmt import (standardize_and_spectrum, mp_support, mp_cdf, trim_interval,
                 ks_against_mp)
from spikes import signal_noise_split


def test_mp_support_formula():
    lm, lp = mp_support(0.25)
    assert lm == pytest.approx((1 - 0.5) ** 2)     # 0.25
    assert lp == pytest.approx((1 + 0.5) ** 2)     # 2.25


def test_mp_cdf_monotone_and_bounded():
    xs = np.linspace(0, 3, 200)
    c = mp_cdf(xs, 0.25)
    assert c[0] == pytest.approx(0.0, abs=1e-6)
    assert c[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(np.diff(c) >= -1e-9)             # non-decreasing


@pytest.mark.parametrize("shape", [(256, 256), (256, 1024)])
def test_gaussian_null_is_well_calibrated(shape):
    """The KEY calibration invariant: an iid-Gaussian matrix run through the
    exact pipeline must yield a small D_p (~<0.03), regardless of square vs
    rectangular. This is what makes any real-weight D_p meaningful."""
    rng = np.random.default_rng(0)
    Ds = []
    for _ in range(15):
        lam, beta, p, N = standardize_and_spectrum(rng.standard_normal(shape))
        Ds.append(ks_against_mp(lam, beta, N, 2)[0])
    assert np.mean(Ds) < 0.03


def test_scale_normalization_puts_spectrum_on_mp_support():
    """lambda = s^2 / N must place the bulk inside [lambda_-, lambda_+]."""
    rng = np.random.default_rng(1)
    lam, beta, p, N = standardize_and_spectrum(rng.standard_normal((256, 1024)))
    lm, lp = mp_support(beta)
    # the vast majority of eigenvalues fall within the MP support
    inside = np.mean((lam >= lm * 0.5) & (lam <= lp * 1.5))
    assert inside > 0.9
    assert beta == pytest.approx(0.25)


def test_orientation_beta_leq_one():
    """beta is always min/max in (0,1], regardless of input orientation."""
    rng = np.random.default_rng(2)
    _, b1, _, _ = standardize_and_spectrum(rng.standard_normal((100, 400)))
    _, b2, _, _ = standardize_and_spectrum(rng.standard_normal((400, 100)))
    assert 0 < b1 <= 1 and 0 < b2 <= 1
    assert b1 == pytest.approx(b2)                 # transpose-invariant


def test_signal_spike_detection_on_planted_signal():
    """A matrix with a planted low-rank spike must show spikes above the bulk."""
    rng = np.random.default_rng(3)
    G = rng.standard_normal((256, 1024))
    G[:, 0] += 8 * G[:, 1]                          # inject a strong direction
    lam, beta, p, N = standardize_and_spectrum(G)
    sn = signal_noise_split(lam, beta)
    assert sn["n_spikes"] >= 1
    assert sn["spike_energy"] > 0


def test_trim_interval_square_uses_hard_edge():
    L, U = trim_interval(1.0, 768, 2)
    assert L > 0                                    # lower edge excluded at 0 (hard edge)
    lm, lp = mp_support(1.0)
    assert U < lp                                   # upper edge trimmed
