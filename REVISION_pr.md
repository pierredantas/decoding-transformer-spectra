# Pattern Recognition revision — added analyses

These scripts extend the pipeline for the PR revision. Each drops into the repo root
and reuses the existing modules (`rmt.py`, `extract.py`, `spikes.py`,
`compression_multi.py`). They make the repo self-contained: `git clone` and run —
no external notebook needed.

| Script | Adds | Run |
|---|---|---|
| `weightwatcher_compare.py` | D_TW vs. WeightWatcher power-law α, per matrix | `python weightwatcher_compare.py bert-base-uncased bert-large-uncased albert-base-v2` |
| `vit_support.py` | Vision-Transformer support (SHA pin + validation) | `python vit_support.py google/vit-base-patch16-224` |
| `compression_aware.py` | Strong SVD baselines: ASVD + FWSVD | `python compression_aware.py bert-base-uncased bert-large-uncased` |
| `spikes_bbp.py` | Spiked-model theory + BBP-informed spike cut | `python spikes_bbp.py` (self-test) |
| `bbp_sensitivity.py` | Spike-count / spectral-allocation sensitivity to the cut | `python bbp_sensitivity.py bert-base-uncased bert-large-uncased` |
| `ablate.py` | Robustness sweep (trim margin, ε floor, FDR q, null convergence) | `python ablate.py bert-base-uncased bert-large-uncased` |
| `positional_control.py` | Position-embedding: smooth-positional vs learned | `python positional_control.py bert-base-uncased` |

`extract.py` also gains a pinned revision for `google/vit-base-patch16-224`, so the
ViT enters the spectral characterization (`run_characterize.py`) on the same footing
as BERT/ALBERT. ViT has no masked-LM head, so — like ALBERT — it is characterized but
excluded from the compression stress-test.

## Full compression sweep (replaces the ad-hoc Colab notebook)
```bash
python compression_multi.py  bert-base-uncased bert-large-uncased   # baseline/gd/uniform/random/spectral
python compression_aware.py  bert-base-uncased bert-large-uncased   # appends asvd/fwsvd
python figure_compression_multi.py                                  # regenerates the figure
```

Notes: `compression_aware.py`'s FWSVD path uses a calibration backward pass; ASVD uses
forward hooks only. Deps: `transformers`, `datasets`, `torch`, `scipy`, `numpy`
(optional `weightwatcher` for the `--use-ww` cross-check in `weightwatcher_compare.py`).
