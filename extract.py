"""Extract 2D weight matrices from HF Transformer checkpoints, grouped by family."""
import re
import numpy as np
from transformers import AutoModel

# HF commit SHAs pinned for exact reproducibility (fetched 2026-07-03).
REVISIONS = {
    "bert-base-uncased":  "86b5e0934494bd15c9632b12f734a8a67f723594",
    "bert-large-uncased": "6da4b6a26a1877e173fca3225479512db81a5e5b",
    "albert-base-v2":     "8e2f239c5f8a2c0f253781ca60135db913e5c80c",
    "google/vit-base-patch16-224": "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
}


def family_of(name):
    """Map a parameter name to a spectral family label, or None to skip."""
    n = name.lower()
    if "layernorm" in n or n.endswith(".bias"):
        return None
    if re.search(r"attention.*(query|key|value)\.weight", n) or \
       re.search(r"attention\.(self\.)?(query|key|value)\.weight", n):
        return "attn_qkv"
    if re.search(r"attention.*(output\.dense|dense)\.weight", n):
        return "attn_out"
    if "intermediate.dense.weight" in n or re.search(r"\.ffn\.weight", n):
        return "ffn_intermediate"
    if re.search(r"(^|\.)output\.dense\.weight", n) or "ffn_output.weight" in n:
        return "ffn_output"
    if "pooler" in n and n.endswith("weight"):
        return "pooler"
    if "word_embeddings" in n:
        return "emb_word"
    if "position_embeddings" in n:
        return "emb_position"
    if "token_type_embeddings" in n:
        return "emb_token_type"
    if "embedding_hidden_mapping_in" in n:
        return "emb_projection"
    return None


def layer_index(name):
    m = re.search(r"\.(?:layer|albert_layers)\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def extract(model_id):
    """Return list of dicts: {name, family, layer, shape, W(np.float64)}."""
    model = AutoModel.from_pretrained(model_id, revision=REVISIONS.get(model_id))
    out = []
    for name, p in model.named_parameters():
        if p.dim() != 2:
            continue
        fam = family_of(name)
        if fam is None:
            continue
        out.append(dict(name=name, family=fam, layer=layer_index(name),
                        shape=tuple(p.shape),
                        W=p.detach().cpu().numpy().astype(np.float64)))
    return out


if __name__ == "__main__":
    import sys
    for mid in (sys.argv[1:] or ["albert-base-v2"]):
        mats = extract(mid)
        from collections import Counter
        c = Counter(m["family"] for m in mats)
        print(f"\n{mid}: {len(mats)} matrices")
        for fam, k in sorted(c.items()):
            ex = next(m for m in mats if m["family"] == fam)
            print(f"  {fam:18s} n={k:3d}  e.g. {ex['shape']}")
