"""B1 (review Weakness 3): add a STRONG training-free SVD-compression baseline to
the stress-test, so "no allocation compresses BERT for free" is established against
the methods actually designed to do it -- not only the naive spectral/uniform/random
controls. Adds two activation/Fisher-aware truncation methods as new CSV `mode`s:

  asvd   -- Activation-aware SVD (Yuan et al. 2023): scale W's input channels by
            their calibration activation RMS, truncate in that basis, unscale.
  fwsvd  -- Fisher-Weighted SVD (Hsu et al., ICLR 2022): weight W's rows by Fisher
            information (accumulated squared gradients on a calibration set),
            truncate in that basis, unweight.

Both reuse compression_multi.py's budgeting, evaluation, and CSV schema so the new
rows drop straight into figure_compression_multi.py alongside the existing methods.
Run AFTER compression_multi.py has written out/compression_multi.csv (append mode):

    python compression_aware.py bert-base-uncased bert-large-uncased

DESIGN NOTES / WHAT TO VERIFY
-----------------------------
* Rank budget: we reuse UNIFORM allocation (plan_ranks(..., "uniform")) at the same
  BUDGETS as the paper, so asvd/fwsvd differ from `uniform` ONLY in the truncation
  basis -- an apples-to-apples "does a smarter basis help?" comparison. (If you want
  activation-aware *allocation* too, that is a separate axis; keep it out of v1.)
* asvd is fully runnable as written (forward hooks, no grad).
* fwsvd needs a backward pass; compression_multi disables grad globally, so we
  re-enable it locally. VERIFY the Fisher accumulation fits in memory for BERT-large
  (it stores one float per weight element per target matrix) and that the calibration
  set / masking matches build_eval so the comparison is fair.
"""
import sys, csv, numpy as np, torch, torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer
from extract import REVISIONS
from compression_multi import (
    MODELS, BUDGETS, SEEDS, is_target, signal_rank,
    plan_ranks, kept_fraction, build_eval, mlm_loss, apply_ranks)

N_CALIB = 64          # calibration sequences for activation/Fisher stats
EPS = 1e-8


# ---------- activation / Fisher statistics via a calibration pass ----------

def _target_linears(model):
    """Map target weight-name -> nn.Linear module (for hooks / Fisher)."""
    out = {}
    for mod_name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and is_target(mod_name + ".weight"):
            out[mod_name + ".weight"] = mod
    return out


def activation_scales(model, calib_ids, calib_attn):
    """ASVD: per-INPUT-channel RMS of each target Linear's input over calibration data.
    Returns {weight_name: np.array of length in_features}."""
    mods = _target_linears(model)
    acc = {k: None for k in mods}
    handles = []

    def mk_hook(key):
        def hook(_m, inp, _out):
            x = inp[0].detach()                       # (..., in_features)
            x = x.reshape(-1, x.shape[-1]).double()
            s = (x * x).sum(0)                         # sum of squares per channel
            acc[key] = s if acc[key] is None else acc[key] + s
        return hook

    name_of = {m: k for k, m in mods.items()}
    for k, m in mods.items():
        handles.append(m.register_forward_hook(mk_hook(k)))
    with torch.no_grad():
        model(input_ids=calib_ids, attention_mask=calib_attn)
    for h in handles:
        h.remove()
    n = calib_ids.numel()
    return {k: np.sqrt((v.cpu().numpy() / n)) + EPS for k, v in acc.items()}


def fisher_scales(model, calib_ids, calib_attn, calib_labels):
    """FWSVD: per-ROW Fisher information of each target weight, approximated by
    accumulated squared gradients on the calibration set. Returns
    {weight_name: np.array of length out_features} (row weights)."""
    mods = _target_linears(model)
    for m in mods.values():
        m.weight.requires_grad_(True)
    fisher = {k: torch.zeros_like(m.weight, dtype=torch.double) for k, m in mods.items()}
    with torch.enable_grad():
        out = model(input_ids=calib_ids, attention_mask=calib_attn, labels=calib_labels)
        model.zero_grad(set_to_none=True)
        out.loss.backward()
        for k, m in mods.items():
            if m.weight.grad is not None:
                fisher[k] += (m.weight.grad.double() ** 2)
    # row Fisher = sum over input dim; row weight = sqrt(I_row)
    return {k: (np.sqrt(v.sum(dim=1).cpu().numpy()) + EPS) for k, v in fisher.items()}


# ---------- weighted truncated-SVD reconstructions ----------

def asvd_truncate(W, col_scale, r):
    """Truncate SVD in the activation-scaled column basis. col_scale: (in_features,)."""
    r = int(np.clip(r, 1, min(W.shape)))
    Ws = W * col_scale[None, :]
    U, s, Vt = np.linalg.svd(Ws, full_matrices=False)
    What_s = (U[:, :r] * s[:r]) @ Vt[:r]
    return What_s / col_scale[None, :]


def fwsvd_truncate(W, row_w, r):
    """Truncate SVD in the Fisher-weighted row basis. row_w: (out_features,)."""
    r = int(np.clip(r, 1, min(W.shape)))
    Ww = W * row_w[:, None]
    U, s, Vt = np.linalg.svd(Ww, full_matrices=False)
    What_w = (U[:, :r] * s[:r]) @ Vt[:r]
    return What_w / row_w[:, None]


# ---------- driver ----------

def apply_ranks_weighted(model, ranks, orig, scales, method):
    sd = model.state_dict()
    for k, r in ranks.items():
        if method == "asvd":
            Wc = asvd_truncate(orig[k], scales[k], r)
        else:  # fwsvd
            Wc = fwsvd_truncate(orig[k], scales[k], r)
        sd[k] = torch.from_numpy(Wc).to(sd[k].dtype)
    model.load_state_dict(sd)


def run_model(mid, writer, methods=("asvd", "fwsvd")):
    tok = AutoTokenizer.from_pretrained(mid, revision=REVISIONS.get(mid))
    model = AutoModelForMaskedLM.from_pretrained(mid, revision=REVISIONS.get(mid)); model.eval()
    orig = {n: p.detach().cpu().numpy().astype(np.float64)
            for n, p in model.named_parameters() if is_target(n)}
    targets = {n: (W.shape[0], W.shape[1], min(W.shape), signal_rank(W)) for n, W in orig.items()}
    evs = [build_eval(tok, s) for s in SEEDS]

    # calibration stats (seed 0's masked batch, reused for both methods for fairness)
    cids, cattn, clabels = build_eval(tok, seed=0, n_seq=N_CALIB)
    scales = {}
    if "asvd" in methods:
        scales["asvd"] = activation_scales(model, cids, cattn)
    if "fwsvd" in methods:
        scales["fwsvd"] = fisher_scales(model, cids, cattn, clabels)
        model.eval()  # fisher pass left grad on; restore eval/no-grad posture
        torch.set_grad_enabled(False)

    def record(method, target, ranks):
        kf = kept_fraction(ranks, targets)
        apply_ranks_weighted(model, ranks, orig, scales[method], method)
        for s, ev in zip(SEEDS, evs):
            writer.writerow(dict(model=mid, mode=method, target_frac=target,
                                 kept_frac=round(kf, 4), seed=s,
                                 mlm_loss=round(mlm_loss(model, ev), 4)))
        # restore full-rank weights before the next budget point
        apply_ranks(model, {k: min(orig[k].shape) for k in orig}, orig)
        print(f"  {method:6s} target={target} kept={kf:.3f}", flush=True)

    for frac in BUDGETS:
        ranks = plan_ranks(targets, frac, "uniform")   # same budget as `uniform`
        for method in methods:
            record(method, frac, ranks)


if __name__ == "__main__":
    mids = sys.argv[1:] or ["bert-base-uncased", "bert-large-uncased"]  # ALBERT/ViT: no usable MLM head
    with open("out/compression_multi.csv", "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "mode", "target_frac", "kept_frac", "seed", "mlm_loss"])
        if fh.tell() == 0:
            w.writeheader()
        for mid in mids:
            print(f"\n########## {mid} ##########", flush=True)
            run_model(mid, w); fh.flush()
    print("appended asvd/fwsvd rows -> out/compression_multi.csv")
