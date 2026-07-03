# Decoding Transformer Spectra — reproducible pipeline

A calibration-verified Random Matrix Theory (RMT) pipeline for the spectra of
Transformer weight matrices (BERT-base, BERT-large, ALBERT), plus an empirical
compression test. Every number comes from real weights; the code is pinned to
exact library versions and Hugging Face model commits.

## Key invariant

An iid-Gaussian matrix run through the exact pipeline yields **D_p ≈ 0.005**
regardless of shape (square vs rectangular). This calibration is what makes any
real-weight statistic interpretable, and it is enforced by the test suite.

## Install

```bash
python -m pip install -r requirements.txt
# For a CPU-only torch build (recommended on CI):
#   pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
```

Pinned environment: Python 3.12, numpy 1.26.4, scipy 1.13.1, torch 2.12.0,
transformers 5.5.4, datasets 5.0.0, matplotlib 3.9.2. Model commits are pinned
in `extract.py` (`REVISIONS`).

## Reproduce

```bash
# 1. Fast invariant tests (seconds, no model download)
python -m pytest tests/ -q

# 2. Spectral characterization (E1/E2) -> out/characterize.csv
python run_characterize.py bert-base-uncased albert-base-v2 bert-large-uncased

# 3. Shape-matched Gaussian null envelopes (E2) -> out/null_envelopes.csv
python nullcontrol.py 200

# 4. Compression validation (E4, BERT-base + WikiText-2) -> out/compression.csv
python compression.py

# 5. Regenerate all figures (E5). FIG_OUT sets the output dir.
FIG_OUT=figs python figures_all.py            # bert_base albert bert_large
```

## What the code does

| File | Role |
|---|---|
| `rmt.py` | Standardize, thin SVD, `λ = s²/max(m,n)` (MP scale), MP support, Tracy–Widom soft-edge trim, hard-edge (β=1), strict KS + KS-TW, bootstrap p-value |
| `extract.py` | Extract 2D weights by family; **pinned model revisions**; excludes degenerate matrices |
| `spikes.py` | Signal/noise split at the MP upper edge (spike count, spike energy) |
| `run_characterize.py` | Per-matrix stats + per-shape bootstrap null → `out/characterize.csv` |
| `nullcontrol.py` | Shape-matched Gaussian null envelopes |
| `compression.py` | Truncated-SVD compression (spectral vs uniform vs random) vs MLM loss |
| `figures.py`, `figures_all.py` | Regenerate manuscript figures from real data |
| `tests/` | Fast invariants (calibration, scale, orientation, spike detection) |

## Findings (summary)

- No Transformer weight matrix is *globally* Marchenko–Pastur, but the **bulk is
  near-MP once outlier spikes are removed**.
- **FFN ≈ attention** in bulk conformity — aspect ratio β does not separate them;
  the structured operators are the pooler and position embeddings.
- **Spectral-guided compression does not beat uniform** (it is worse than random);
  BERT weights are not low-rank and do not compress for free without fine-tuning.

## Continuous integration

- **`.github/workflows/ci.yml`** — every push/PR: invariant tests + a small
  end-to-end run on ALBERT; uploads CSV/figures as artifacts (fast).
- **`.github/workflows/full-reproduction.yml`** — manual (`workflow_dispatch`):
  full pipeline incl. optional BERT-large + compression; uploads all artifacts.

This directory is self-contained: use it as the repository root
(`git init` here), or nest it inside the paper repo and adjust the workflow paths.
