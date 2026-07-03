"""Signal/noise decomposition of a spectrum relative to the MP bulk.

Spikes = eigenvalues above the MP upper edge lambda_+ (structured signal).
Bulk   = eigenvalues within [lambda_-, lambda_+] (candidate MP noise).
"""
import numpy as np
from rmt import mp_support


def signal_noise_split(lam, beta):
    """Return dict of spike/bulk descriptors for eigenvalues lam at aspect beta.

    n_spikes         : count of eigenvalues > lambda_+
    spike_frac       : fraction of eigenvalues that are spikes
    spike_energy     : sum(lambda_i > lambda_+) / sum(all lambda)  (energy above bulk)
    signal_rank      : n_spikes (interpretable as retained rank if we keep spikes)
    bulk             : eigenvalues in [lambda_-, lambda_+]
    """
    lam = np.asarray(lam, dtype=np.float64)
    lm, lp = mp_support(beta)
    spikes = lam[lam > lp]
    bulk = lam[(lam >= lm) & (lam <= lp)]
    total = float(lam.sum()) if lam.sum() > 0 else 1.0
    return dict(
        n_spikes=int(spikes.size),
        spike_frac=float(spikes.size) / lam.size,
        spike_energy=float(spikes.sum()) / total,
        signal_rank=int(spikes.size),
        n_below=int((lam < lm).sum()),
        bulk=bulk,
    )
