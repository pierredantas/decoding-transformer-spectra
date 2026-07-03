"""Core validation: per-matrix / per-family KS statistics on real weights."""
import sys, csv, numpy as np
from collections import defaultdict
from extract import extract
from rmt import standardize_and_spectrum, ks_against_mp

MIN_EIG = 10   # skip degenerate matrices (e.g. token-type 2xN) for KS

def run(model_id):
    mats = extract(model_id)
    rows = []
    for m in mats:
        lam, beta, p, N = standardize_and_spectrum(m["W"])
        if p < MIN_EIG:
            rows.append(dict(model=model_id, **{k: m[k] for k in ("name","family","layer","shape")},
                             beta=beta, p=p, N=N, Dstrict=np.nan, DTW1=np.nan,
                             DTW2=np.nan, DTW3=np.nan, degenerate=True))
            continue
        ds, _ = ks_against_mp(lam, beta, N, None)
        d1, _ = ks_against_mp(lam, beta, N, 1)
        d2, _ = ks_against_mp(lam, beta, N, 2)
        d3, _ = ks_against_mp(lam, beta, N, 3)
        rows.append(dict(model=model_id, name=m["name"], family=m["family"],
                         layer=m["layer"], shape=m["shape"], beta=beta, p=p, N=N,
                         Dstrict=ds, DTW1=d1, DTW2=d2, DTW3=d3, degenerate=False))
    return rows

def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        if not r["degenerate"]:
            by[r["family"]].append(r)
    print(f"\n=== {rows[0]['model']} : per-family KS (mean +/- sd) ===")
    print(f"{'family':18s} {'beta':>5} {'n':>3}  {'D_strict':>16} {'D_TW(c=2)':>16}")
    order = ["ffn_intermediate","ffn_output","attn_qkv","attn_out","pooler",
             "emb_word","emb_position","emb_projection"]
    for fam in order:
        rs = by.get(fam)
        if not rs: continue
        beta = np.mean([r["beta"] for r in rs])
        ds = np.array([r["Dstrict"] for r in rs]); dt = np.array([r["DTW2"] for r in rs])
        print(f"{fam:18s} {beta:5.2f} {len(rs):3d}  "
              f"{np.nanmean(ds):7.4f} +/-{np.nanstd(ds):6.4f}  "
              f"{np.nanmean(dt):7.4f} +/-{np.nanstd(dt):6.4f}")
    deg = [r for r in rows if r["degenerate"]]
    if deg:
        print(f"  [skipped degenerate p<{MIN_EIG}: " +
              ", ".join(f"{r['family']}{r['shape']}" for r in deg) + "]")

if __name__ == "__main__":
    allrows = []
    for mid in (sys.argv[1:] or ["bert-base-uncased","albert-base-v2"]):
        rows = run(mid); summarize(rows); allrows += rows
    keys = ["model","name","family","layer","shape","beta","p","N",
            "Dstrict","DTW1","DTW2","DTW3","degenerate"]
    with open("out/core_ks.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in allrows: w.writerow({k:r[k] for k in keys})
    print("\nwrote out/core_ks.csv")
