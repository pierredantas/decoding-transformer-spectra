"""E2: shape-matched iid Gaussian null envelopes (honest replacement for the
synthetic 'controls' figures). For each distinct matrix shape, characterize the
null distribution of D_strict / D_TW under a true MP null."""
import sys, csv, numpy as np
from rmt import standardize_and_spectrum, ks_against_mp

# distinct (p, N) shapes across BERT-base/large + ALBERT weight families
SHAPES = {
    "attn 768":   (768, 768),
    "ffn 768":    (768, 3072),
    "attn 1024":  (1024, 1024),
    "ffn 1024":   (1024, 4096),
    "emb 768x30k":(768, 30522),
    "albert proj":(128, 768),
}

def null_envelope(p, N, K=200, seed=0):
    rng = np.random.default_rng(seed)
    ds, dt = [], []
    for _ in range(K):
        lam, beta, pp, NN = standardize_and_spectrum(rng.standard_normal((p, N)))
        ds.append(ks_against_mp(lam, beta, NN, None)[0])
        dt.append(ks_against_mp(lam, beta, NN, 2)[0])
    ds, dt = np.array(ds), np.array(dt)
    q = lambda a: (float(np.mean(a)), float(np.percentile(a, 95)), float(np.percentile(a, 99)))
    return q(ds), q(dt)

if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rows = []
    print(f"{'shape':14s} {'beta':>5} {'Dstrict mean/95/99':>26} {'DTW mean/95/99':>26}")
    for tag, (p, N) in SHAPES.items():
        (dsm,ds95,ds99),(dtm,dt95,dt99) = null_envelope(p, N, K=K)
        beta = min(p,N)/max(p,N)
        print(f"{tag:14s} {beta:5.2f}  {dsm:.4f}/{ds95:.4f}/{ds99:.4f}   "
              f"{dtm:.4f}/{dt95:.4f}/{dt99:.4f}")
        rows.append(dict(shape=tag, p=p, N=N, beta=beta,
                         Dstrict_mean=dsm, Dstrict_p95=ds95, Dstrict_p99=ds99,
                         DTW_mean=dtm, DTW_p95=dt95, DTW_p99=dt99))
    with open("out/null_envelopes.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print("\nwrote out/null_envelopes.csv")
