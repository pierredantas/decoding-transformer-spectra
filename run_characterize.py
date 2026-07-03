"""E1+E2: full real-weight spectral characterization.

Per matrix: beta, p, N, D_strict, D_TW(c=2), n_spikes, spike_energy, and a
bootstrap p-value obtained by comparing D_TW to a per-SHAPE null distribution
(computed once per distinct (p,N) shape, since the MP null depends only on shape).
Skips degenerate matrices (p<MIN_EIG). Aggregates per family with bootstrap CIs.
"""
import sys, csv, numpy as np
from collections import defaultdict
from extract import extract
from rmt import standardize_and_spectrum, ks_against_mp
from spikes import signal_noise_split

MIN_EIG = 10
def nboot_for(N):
    return 60 if N > 8000 else 200

_NULL_CACHE = {}
def shape_null(p, N, seed=12345):
    """Null distribution of (D_strict, D_TW) for iid Gaussian of shape (p,N).
    Cached per shape; returns sorted arrays for p-value lookup."""
    key = (p, N)
    if key in _NULL_CACHE: return _NULL_CACHE[key]
    rng = np.random.default_rng(seed)
    K = nboot_for(N); ds, dt = [], []
    for _ in range(K):
        lam, b, pp, NN = standardize_and_spectrum(rng.standard_normal((p, N)))
        ds.append(ks_against_mp(lam, b, NN, None)[0])
        dt.append(ks_against_mp(lam, b, NN, 2)[0])
    val = (np.sort(ds), np.sort(dt))
    _NULL_CACHE[key] = val
    print(f"    [null shape ({p},{N}) K={K}: DTW mean={np.mean(dt):.4f} "
          f"p95={np.percentile(dt,95):.4f}]", flush=True)
    return val

def pval(null_sorted, obs):
    return float((np.sum(null_sorted >= obs) + 1) / (null_sorted.size + 1))

def characterize(model_id):
    mats = extract(model_id)
    mats.sort(key=lambda m: max(m["shape"]))
    rows = []
    for m in mats:
        lam, beta, p, N = standardize_and_spectrum(m["W"])
        base = dict(model=model_id, name=m["name"], family=m["family"],
                    layer=m["layer"], shape=str(m["shape"]), beta=beta, p=p, N=N)
        if p < MIN_EIG:
            rows.append({**base, "Dstrict": np.nan, "DTW2": np.nan,
                         "n_spikes": np.nan, "spike_energy": np.nan,
                         "boot_p_TW": np.nan, "degenerate": True})
            continue
        ds, _ = ks_against_mp(lam, beta, N, None)
        dt, _ = ks_against_mp(lam, beta, N, 2)
        sn = signal_noise_split(lam, beta)
        nd_strict, nd_tw = shape_null(p, N)
        bp = pval(nd_tw, dt)
        rows.append({**base, "Dstrict": ds, "DTW2": dt,
                     "n_spikes": sn["n_spikes"], "spike_energy": sn["spike_energy"],
                     "boot_p_TW": bp, "degenerate": False})
        print(f"  {m['family']:16s} L{m['layer']:>2} beta={beta:4.2f} "
              f"Dstrict={ds:.3f} DTW={dt:.3f} spikes={sn['n_spikes']:>3} "
              f"Espike={sn['spike_energy']:.3f} boot_p={bp:.3f}", flush=True)
    return rows

def boot_ci(vals, n=2000, seed=1):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0: return (np.nan, np.nan, np.nan)
    if vals.size == 1: return (vals[0], vals[0], vals[0])
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, vals.size, replace=True).mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        if not r["degenerate"]: by[r["family"]].append(r)
    print(f"\n=== {rows[0]['model']}: per-family (mean [95% CI]) ===")
    print(f"{'family':16s} {'beta':>4} {'n':>3} {'D_strict [CI]':>22} "
          f"{'D_TW [CI]':>22} {'spikes':>7} {'Espike':>7} {'acc(boot p>.05)':>15}")
    order = ["ffn_intermediate","ffn_output","attn_qkv","attn_out","pooler",
             "emb_word","emb_position","emb_projection"]
    for fam in [f for f in order if f in by] + [f for f in by if f not in order]:
        rs = by[fam]; beta = np.mean([r["beta"] for r in rs])
        m,l,u = boot_ci([r["Dstrict"] for r in rs])
        m2,l2,u2 = boot_ci([r["DTW2"] for r in rs])
        sp = np.nanmean([r["n_spikes"] for r in rs])
        en = np.nanmean([r["spike_energy"] for r in rs])
        acc = np.mean([r["boot_p_TW"] > 0.05 for r in rs])
        print(f"{fam:16s} {beta:4.2f} {len(rs):3d} "
              f"{m:6.3f}[{l:.3f},{u:.3f}] {m2:6.3f}[{l2:.3f},{u2:.3f}] "
              f"{sp:7.1f} {en:7.3f} {acc:15.2f}")

if __name__ == "__main__":
    allrows = []
    for mid in (sys.argv[1:] or ["bert-base-uncased", "albert-base-v2"]):
        print(f"\n########## {mid} ##########", flush=True)
        rows = characterize(mid); summarize(rows); allrows += rows
        keys = list(allrows[0].keys())
        with open("out/characterize.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in allrows: w.writerow(r)
    print("\nwrote out/characterize.csv")
