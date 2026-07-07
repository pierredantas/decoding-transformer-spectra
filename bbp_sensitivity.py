"""B2 sensitivity driver (review Weakness 4 / Question 3): does the spike threshold
choice -- bare MP edge lambda_+ vs. the BBP-informed TW-margined cut -- change the
conclusions that depend on spike counts?

Two things depend on the cut, so we test both:

  (1) Table 3 quantities: per-family n_spikes and spike_energy (E_spike), and the
      "pooler / position embeddings are the structured operators" claim.
  (2) The Section 6 SPECTRAL COMPRESSION ALLOCATION, which allocates per-layer rank
      in proportion to each layer's spike count (signal_rank). If the allocation is
      stable across the two cuts, the compression result is robust to the referee's
      concern; if it drifts, that drift must be reported.

Drop into the repo root and run (needs transformers to load the pinned weights):
    python bbp_sensitivity.py bert-base-uncased bert-large-uncased albert-base-v2
Writes out/bbp_sensitivity.csv and prints two summary tables. The manuscript
deliverable is a supplementary table built from these numbers plus one sentence:
either "spike counts and the spectral allocation are stable to the edge choice
(max per-layer rank change = X%)", or a quantified statement of the drift.
"""
import sys, csv, os, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from extract import extract
from rmt import standardize_and_spectrum
from spikes_bbp import signal_noise_split_bbp
from compression_multi import is_target, plan_ranks, kept_fraction, BUDGETS

MIN_EIG = 10
C_ALPHA = 2          # BBP-informed cut uses the same TW constant as the D_TW trim


def characterization_sensitivity(model_id, rows):
    """Per-matrix spike count / energy under naive (lambda_+) vs BBP cut."""
    per_family = defaultdict(lambda: defaultdict(list))
    for m in extract(model_id):
        lam, beta, p, N = standardize_and_spectrum(m["W"])
        if p < MIN_EIG:
            continue
        naive = signal_noise_split_bbp(lam, beta, N, c_alpha=None)
        bbp = signal_noise_split_bbp(lam, beta, N, c_alpha=C_ALPHA)
        rows.append(dict(model=model_id, name=m["name"], family=m["family"],
                         beta=round(beta, 4),
                         n_spikes_naive=naive["n_spikes"], n_spikes_bbp=bbp["n_spikes"],
                         near_edge=bbp["near_edge_spikes"],
                         Espike_naive=round(naive["spike_energy"], 4),
                         Espike_bbp=round(bbp["spike_energy"], 4)))
        f = per_family[m["family"]]
        f["ns_n"].append(naive["n_spikes"]); f["ns_b"].append(bbp["n_spikes"])
        f["en_n"].append(naive["spike_energy"]); f["en_b"].append(bbp["spike_energy"])
    print(f"\n=== {model_id}: characterization sensitivity (naive lambda_+ -> BBP cut) ===")
    print(f"{'family':16s} {'n_spikes':>16} {'E_spike':>18}")
    print(f"{'':16s} {'naive -> bbp':>16} {'naive -> bbp':>18}")
    for fam, f in sorted(per_family.items()):
        print(f"{fam:16s} {np.mean(f['ns_n']):6.1f} -> {np.mean(f['ns_b']):6.1f}   "
              f"{np.mean(f['en_n']):7.3f} -> {np.mean(f['en_b']):7.3f}")


def allocation_sensitivity(model_id):
    """Spectral rank allocation under both cuts, at the paper's budgets."""
    mats = {m["name"]: m["W"] for m in extract(model_id) if is_target(m["name"])}

    def targets_for(c_alpha):
        t = {}
        for name, W in mats.items():
            lam, beta, p, N = standardize_and_spectrum(W)
            sig = max(1, signal_noise_split_bbp(lam, beta, N, c_alpha)["n_spikes"])
            t[name] = (W.shape[0], W.shape[1], min(W.shape), sig)
        return t

    t_naive, t_bbp = targets_for(None), targets_for(C_ALPHA)
    print(f"\n=== {model_id}: spectral ALLOCATION sensitivity ===")
    print(f"{'budget':>7} {'kept naive':>11} {'kept bbp':>9} "
          f"{'max|dRank|':>11} {'mean|dRank|':>12} {'rank rho':>9}")
    out = []
    for frac in BUDGETS:
        r_naive = plan_ranks(t_naive, frac, "spectral")
        r_bbp = plan_ranks(t_bbp, frac, "spectral")
        keys = list(r_naive)
        d = np.array([r_bbp[k] - r_naive[k] for k in keys], float)
        rho, _ = spearmanr([r_naive[k] for k in keys], [r_bbp[k] for k in keys])
        row = dict(model=model_id, budget=frac,
                   kept_naive=round(kept_fraction(r_naive, t_naive), 4),
                   kept_bbp=round(kept_fraction(r_bbp, t_bbp), 4),
                   max_abs_drank=int(np.abs(d).max()),
                   mean_abs_drank=round(float(np.abs(d).mean()), 3),
                   rank_spearman=round(float(rho), 4))
        out.append(row)
        print(f"{frac:7.2f} {row['kept_naive']:11.3f} {row['kept_bbp']:9.3f} "
              f"{row['max_abs_drank']:11d} {row['mean_abs_drank']:12.3f} "
              f"{row['rank_spearman']:9.3f}")
    print("  interpretation: rank rho ~ 1 and small mean|dRank| => the spectral")
    print("  compression allocation is robust to the spike-threshold choice.")
    return out


if __name__ == "__main__":
    mids = sys.argv[1:] or ["bert-base-uncased", "bert-large-uncased", "albert-base-v2"]
    char_rows, alloc_rows = [], []
    for mid in mids:
        characterization_sensitivity(mid, char_rows)
        if mid != "albert-base-v2":          # allocation test mirrors the compression set
            alloc_rows += allocation_sensitivity(mid)
    os.makedirs("out", exist_ok=True)
    with open("out/bbp_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(char_rows[0].keys()))
        w.writeheader(); w.writerows(char_rows)
    with open("out/bbp_allocation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(alloc_rows[0].keys()))
        w.writeheader(); w.writerows(alloc_rows)
    print("\nwrote out/bbp_sensitivity.csv and out/bbp_allocation.csv")
