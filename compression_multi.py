"""E4 (extended): compression validation across models, with a Gavish-Donoho
optimal-hard-threshold baseline and seed repetitions for error bands.

Compares four per-layer rank allocations at matched parameter budgets:
  spectral  -- rank proportional to spectral signal-rank (spike count)
  uniform   -- same rank fraction for every layer
  random    -- random per-layer weights (control)
plus GAVISH-DONOHO (gd), a principled per-matrix optimal hard threshold
[Gavish & Donoho 2014] that selects each layer's rank from the singular-value
distribution under an unknown-noise model (its own natural budget, one point).

Downstream metric: masked-LM loss on a WikiText-2 sample (no fine-tuning),
averaged over several masking seeds. Writes out/compression_multi.csv (long form).
"""
import sys, csv, re, numpy as np, torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from rmt import standardize_and_spectrum
from spikes import signal_noise_split
from extract import REVISIONS

torch.set_grad_enabled(False)
MODELS = ["bert-base-uncased", "bert-large-uncased", "albert-base-v2"]
BUDGETS = [0.7, 0.5, 0.3]
MODES = ["spectral", "uniform", "random"]
SEEDS = [0, 1, 2]

# 6 linear ops per encoder layer, for BERT and ALBERT naming
_TARGET = re.compile(
    r"(attention\.(self\.)?(query|key|value)\.weight)"
    r"|(attention\.(output\.)?dense\.weight)"
    r"|(intermediate\.dense\.weight)|(^|\.)output\.dense\.weight"
    r"|(\.ffn\.weight)|(ffn_output\.weight)")
def is_target(name):
    n = name.lower()
    if "embedding" in n or "pooler" in n or "layernorm" in n:
        return False
    return ("encoder.layer." in n or "albert_layers" in n) and bool(_TARGET.search(n))

def signal_rank(W):
    lam, beta, p, N = standardize_and_spectrum(W)
    return max(1, signal_noise_split(lam, beta)["n_spikes"])

def _mp_median(beta, n=200000):
    lm, lp = (1 - np.sqrt(beta))**2, (1 + np.sqrt(beta))**2
    x = np.linspace(max(lm, 1e-9), lp, n)          # avoid 1/sqrt(lambda) singularity at 0
    d = np.sqrt(np.clip((lp - x) * (x - lm), 0, None)) / (2 * np.pi * beta * x)
    c = np.cumsum(d) * (x[1] - x[0]); c /= c[-1]
    return float(np.interp(0.5, c, x))

def gd_rank(W):
    """Gavish-Donoho optimal hard threshold (unknown noise): keep s_i > omega*median(s)."""
    m, n = W.shape; beta = min(m, n) / max(m, n)
    s = np.linalg.svd(W, compute_uv=False)
    lam = np.sqrt(2 * (beta + 1) + 8 * beta / ((beta + 1) + np.sqrt(beta**2 + 14 * beta + 1)))
    omega = lam / np.sqrt(_mp_median(beta))
    tau = omega * np.median(s)
    return max(1, int(np.sum(s > tau)))

def truncate(W, r):
    r = int(np.clip(r, 1, min(W.shape)))
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]

def plan_ranks(targets, kept_frac, mode):
    full = sum((m + n) * R for (m, n, R, _) in targets.values())
    if mode == "uniform":
        return {k: max(1, round(kept_frac * R)) for k, (m, n, R, _) in targets.items()}
    if mode == "random":
        rng = np.random.default_rng(7)
        w = {k: rng.uniform(0.2, 1.0) for k in targets}
    else:  # spectral
        w = {k: sig for k, (m, n, R, sig) in targets.items()}
    denom = sum((m + n) * w[k] for k, (m, n, R, _) in targets.items())
    alpha = kept_frac * full / denom
    return {k: int(np.clip(round(alpha * w[k]), 1, R)) for k, (m, n, R, _) in targets.items()}

def kept_fraction(ranks, targets):
    kept = sum((m + n) * ranks[k] for k, (m, n, R, _) in targets.items())
    full = sum((m + n) * R for (m, n, R, _) in targets.values())
    return kept / full

def build_eval(tok, seed, n_seq=96, maxlen=128):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    txt = [t for t in ds["text"] if len(t.split()) > 20][:n_seq]
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=maxlen, padding="max_length")
    rng = np.random.default_rng(seed)
    ids, attn = enc["input_ids"].clone(), enc["attention_mask"]
    labels = torch.full_like(ids, -100)
    special = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
    for i in range(ids.size(0)):
        cand = [j for j in range(ids.size(1)) if attn[i, j] == 1 and ids[i, j].item() not in special]
        for j in rng.choice(cand, max(1, int(0.15 * len(cand))), replace=False):
            labels[i, j] = ids[i, j]; ids[i, j] = tok.mask_token_id
    return ids, attn, labels

def mlm_loss(model, ev, bs=16):
    ids, attn, labels = ev; tot, ntok = 0.0, 0
    for i in range(0, ids.size(0), bs):
        lg = model(input_ids=ids[i:i+bs], attention_mask=attn[i:i+bs]).logits
        m = labels[i:i+bs] != -100
        tot += torch.nn.functional.cross_entropy(lg[m], labels[i:i+bs][m], reduction="sum").item()
        ntok += int(m.sum())
    return tot / ntok

def apply_ranks(model, ranks, orig):
    sd = model.state_dict()
    for k, r in ranks.items():
        sd[k] = torch.from_numpy(truncate(orig[k], r)).to(sd[k].dtype)
    model.load_state_dict(sd)

def run_model(mid, writer):
    tok = AutoTokenizer.from_pretrained(mid, revision=REVISIONS.get(mid))
    model = AutoModelForMaskedLM.from_pretrained(mid, revision=REVISIONS.get(mid)); model.eval()
    orig = {n: p.detach().cpu().numpy().astype(np.float64)
            for n, p in model.named_parameters() if is_target(n)}
    targets = {n: (W.shape[0], W.shape[1], min(W.shape), signal_rank(W)) for n, W in orig.items()}
    gd = {n: gd_rank(W) for n, W in orig.items()}
    evs = [build_eval(tok, s) for s in SEEDS]
    print(f"[{mid}] {len(targets)} target matrices; GD kept={kept_fraction(gd,targets):.3f}", flush=True)

    def record(mode, target, ranks):
        kf = kept_fraction(ranks, targets); apply_ranks(model, ranks, orig)
        for s, ev in zip(SEEDS, evs):
            L = mlm_loss(model, ev)
            writer.writerow(dict(model=mid, mode=mode, target_frac=target,
                                 kept_frac=round(kf, 4), seed=s, mlm_loss=round(L, 4)))
        print(f"  {mode:9s} target={target} kept={kf:.3f}", flush=True)

    # baseline (full rank)
    record("baseline", 1.0, {k: min(orig[k].shape) for k in orig})
    # Gavish-Donoho natural operating point
    record("gd", -1.0, gd)
    # budget sweeps
    for frac in BUDGETS:
        for mode in MODES:
            record(mode, frac, plan_ranks(targets, frac, mode))

if __name__ == "__main__":
    mids = sys.argv[1:] or MODELS
    with open("out/compression_multi.csv", "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model","mode","target_frac","kept_frac","seed","mlm_loss"])
        if fh.tell() == 0: w.writeheader()
        for mid in mids:
            run_model(mid, w); fh.flush()
    print("done ->", ", ".join(mids))
