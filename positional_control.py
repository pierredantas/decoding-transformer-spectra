"""W6 / Q7 control: is the position embedding's high D_TW "learned signal", or just
the smooth, deterministic structure of an ordered position index?

The paper flags the position embeddings as the most structured operators (largest
D_TW, highest spike energy). A referee objects that learned ABSOLUTE position
embeddings are known to vary smoothly with position, so a high deviation could be
deterministic-positional rather than task-learned. This control quantifies exactly
that: of the spike energy, how much lies in position-profiles that a low-frequency
(smooth) basis explains?

Method. Orient the position embedding P (T positions x H hidden) with positions as
the p=min variables (true for BERT: 512<768; ViT: 197<768), standardize rows, take
the SVD. The eigenvectors of the sample covariance are then the LEFT singular
vectors u_k in R^T -- one profile over the position index per spike. We project each
spike's u_k onto a low-frequency DCT-II basis over positions and report the
spike-energy-weighted fraction captured. Fraction ~1 => the "structure" is smooth
positional; fraction ~ n_low/T => it is not explained by smoothness.

Runnable standalone self-test (no model needed):  python positional_control.py
Against real weights:  python positional_control.py bert-base-uncased google/vit-base-patch16-224
"""
import sys, numpy as np

EPS = 1e-12


def _mp_upper(beta):                        # mirrors rmt.mp_support upper edge
    return (1.0 + np.sqrt(beta)) ** 2


def low_freq_basis(T, n_low):
    """Orthonormal DCT-II cosine basis for the lowest n_low frequencies over T points."""
    t = np.arange(T)[:, None]
    k = np.arange(n_low)[None, :]
    Phi = np.cos(np.pi * (t + 0.5) * k / T)      # (T, n_low)
    Phi /= np.linalg.norm(Phi, axis=0, keepdims=True)
    return Phi


def analyze(P, n_low_frac=0.10):
    """P: (T, H) position embedding, positions on axis 0 (require T <= H).
    Returns dict with n_spikes and the spike-energy-weighted low-frequency fraction."""
    P = np.asarray(P, float)
    T, H = P.shape
    if T > H:
        raise ValueError(f"expected positions (axis 0) to be the smaller dim; got {P.shape}")
    # standardize each position across hidden dims (pipeline convention, positions=variables)
    mu = P.mean(1, keepdims=True); sd = np.maximum(P.std(1, ddof=0, keepdims=True), EPS)
    Xs = (P - mu) / sd
    U, s, _ = np.linalg.svd(Xs, full_matrices=False)   # U: (T, T) profiles over position
    lam = (s ** 2) / H                                 # MP scale, N = H samples
    beta = T / H
    lp = _mp_upper(beta)
    spike = lam > lp
    n_low = max(2, int(round(n_low_frac * T)))
    Phi = low_freq_basis(T, n_low)
    if spike.sum() == 0:
        return dict(n_spikes=0, n_low=n_low, ew_smoothness=np.nan, per_spike=[])
    fr = []
    for k in np.where(spike)[0]:
        u = U[:, k]                                    # unit-norm position profile
        fr.append(float(np.sum((Phi.T @ u) ** 2)))     # energy in low-freq subspace
    fr = np.array(fr); w = lam[spike]
    return dict(n_spikes=int(spike.sum()), n_low=n_low,
                ew_smoothness=float(np.sum(w * fr) / np.sum(w)),
                per_spike=list(zip(np.round(fr, 3), np.round(w, 3))))


def _from_model(model_id):
    from transformers import AutoModel
    from extract import REVISIONS
    model = AutoModel.from_pretrained(model_id, revision=REVISIONS.get(model_id))
    for name, p in model.named_parameters():
        if "position_embeddings" not in name:
            continue
        W = p.detach().cpu().numpy().astype(np.float64)
        W = W[0] if W.ndim == 3 else W                 # ViT: (1,T,H) -> (T,H)
        if W.ndim != 2:
            continue
        r = analyze(W)
        print(f"  {model_id} {name} shape={W.shape} spikes={r['n_spikes']} "
              f"n_low={r['n_low']} energy-wtd low-freq fraction={r['ew_smoothness']:.3f}")


def _selftest():
    rng = np.random.default_rng(0)
    T, H, r = 128, 768, 8
    Phi = low_freq_basis(T, r)                          # SMOOTH position profiles
    smooth = Phi @ (rng.standard_normal((r, H)) * 6.0) + 0.1 * rng.standard_normal((T, H))
    Q, _ = np.linalg.qr(rng.standard_normal((T, r)))    # ROUGH random profiles
    rough = Q @ (rng.standard_normal((r, H)) * 6.0) + 0.1 * rng.standard_normal((T, H))
    a, b = analyze(smooth), analyze(rough)
    print(f"  smooth-profile matrix: spikes={a['n_spikes']}  low-freq fraction={a['ew_smoothness']:.3f}")
    print(f"  rough-profile  matrix: spikes={b['n_spikes']}  low-freq fraction={b['ew_smoothness']:.3f}")
    assert a["ew_smoothness"] > 0.8, a["ew_smoothness"]
    assert b["ew_smoothness"] < 0.30, b["ew_smoothness"]
    print("  OK: control separates deterministic-smooth from learned-rough structure.")


if __name__ == "__main__":
    mids = sys.argv[1:]
    if not mids:
        print("self-test (synthetic):")
        _selftest()
    else:
        for mid in mids:
            _from_model(mid)
