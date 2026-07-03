"""E4: compression validation.

Tests the paper's central practical claim empirically: does spectral structure
predict compressibility? For each encoder linear weight we compute its spectral
signal-rank (# eigenvalues above the MP bulk). We then compress all encoder
weights by truncated SVD to a target parameter budget under two allocations:

  - spectral-guided: per-layer rank proportional to its signal-rank
  - uniform:         per-layer rank = same fraction of full rank

and measure masked-LM loss on a WikiText-2 sample. If spectral-guided achieves
lower loss at matched budget, spectral structure genuinely guides compression.
"""
import sys, csv, numpy as np, torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from rmt import standardize_and_spectrum
from spikes import signal_noise_split
from extract import REVISIONS

MODEL = "bert-base-uncased"
REVISION = REVISIONS.get(MODEL)
torch.set_grad_enabled(False)

# ---- target weights: the 6 linear ops per encoder layer ----
def is_target(name):
    return name.endswith(".weight") and "encoder.layer." in name and any(
        k in name for k in ["attention.self.query","attention.self.key",
        "attention.self.value","attention.output.dense","intermediate.dense",
        "output.dense"]) and "LayerNorm" not in name

def signal_rank(W):
    lam, beta, p, N = standardize_and_spectrum(W)
    return max(1, signal_noise_split(lam, beta)["n_spikes"])

def truncate(W, r):
    r = int(np.clip(r, 1, min(W.shape)))
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]

def plan_ranks(targets, kept_frac, mode):
    """Return {name: rank}. targets: {name:(m,n,full_rank,signal)}.
    Budget = kept_frac * (sum of full-SVD params) across targets."""
    full_params = sum((m + n) * R for (m, n, R, _) in targets.values())
    budget = kept_frac * full_params
    if mode == "uniform":
        f = kept_frac
        return {nm: max(1, round(f * R)) for nm, (m, n, R, _) in targets.items()}
    if mode == "random":                     # control: random per-layer weights
        rng = np.random.default_rng(7)
        w = {nm: rng.uniform(0.2, 1.0) for nm in targets}
        denom = sum((m + n) * w[nm] for nm, (m, n, R, _) in targets.items())
        alpha = budget / denom
        return {nm: int(np.clip(round(alpha * w[nm]), 1, R))
                for nm, (m, n, R, _) in targets.items()}
    # spectral-guided: rank_i ∝ signal_i, scaled to hit param budget
    denom = sum((m + n) * sig for (m, n, R, sig) in targets.values())
    alpha = budget / denom
    return {nm: int(np.clip(round(alpha * sig), 1, R))
            for nm, (m, n, R, sig) in targets.items()}

def kept_fraction(ranks, targets):
    kept = sum((m + n) * ranks[nm] for nm, (m, n, R, _) in targets.items())
    full = sum((m + n) * R for (m, n, R, _) in targets.values())
    return kept / full

def build_eval(tok, n_seq=200, maxlen=128, seed=0):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    txt = [t for t in ds["text"] if len(t.split()) > 20][:n_seq]
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=maxlen,
              padding="max_length")
    rng = np.random.default_rng(seed)
    ids, attn = enc["input_ids"].clone(), enc["attention_mask"]
    labels = torch.full_like(ids, -100)
    special = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
    for i in range(ids.size(0)):
        cand = [j for j in range(ids.size(1))
                if attn[i, j] == 1 and ids[i, j].item() not in special]
        k = max(1, int(0.15 * len(cand)))
        for j in rng.choice(cand, k, replace=False):
            labels[i, j] = ids[i, j]; ids[i, j] = tok.mask_token_id
    return ids, attn, labels

def mlm_loss(model, ids, attn, labels, bs=16):
    tot, ntok = 0.0, 0
    for i in range(0, ids.size(0), bs):
        out = model(input_ids=ids[i:i+bs], attention_mask=attn[i:i+bs])
        lg = out.logits
        m = labels[i:i+bs] != -100
        loss = torch.nn.functional.cross_entropy(
            lg[m], labels[i:i+bs][m], reduction="sum")
        tot += loss.item(); ntok += int(m.sum())
    return tot / ntok

def apply_ranks(model, ranks, orig):
    sd = model.state_dict()
    for nm, r in ranks.items():
        sd[nm] = torch.from_numpy(truncate(orig[nm], r)).to(sd[nm].dtype)
    model.load_state_dict(sd)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForMaskedLM.from_pretrained(MODEL, revision=REVISION); model.eval()
    orig = {n: p.detach().cpu().numpy().astype(np.float64)
            for n, p in model.named_parameters() if is_target(n)}
    targets = {n: (W.shape[0], W.shape[1], min(W.shape), signal_rank(W))
               for n, W in orig.items()}
    print(f"{len(targets)} target matrices; signal-rank range "
          f"{min(t[3] for t in targets.values())}–{max(t[3] for t in targets.values())}")
    ids, attn, labels = build_eval(tok, n_seq=120)
    base = mlm_loss(model, ids, attn, labels)
    print(f"baseline MLM loss (uncompressed): {base:.4f}", flush=True)

    rows = [dict(mode="baseline", target_frac=1.0, kept_frac=1.0, mlm_loss=base)]
    for frac in [0.9, 0.8, 0.7, 0.6, 0.5]:
        for mode in ["spectral", "uniform", "random"]:
            ranks = plan_ranks(targets, frac, mode)
            kf = kept_fraction(ranks, targets)
            apply_ranks(model, ranks, orig)
            L = mlm_loss(model, ids, attn, labels)
            rows.append(dict(mode=mode, target_frac=frac, kept_frac=round(kf, 4),
                             mlm_loss=round(L, 4)))
            print(f"  {mode:9s} target={frac:.2f} kept={kf:.3f} MLM_loss={L:.4f}")
    with open("out/compression.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode","target_frac","kept_frac","mlm_loss"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print("wrote out/compression.csv")

if __name__ == "__main__":
    main()
