"""A2 (review Weakness 2 / Question 6): add a Vision Transformer to the spectral
characterization so the paper's "Transformer" claims are not encoder-NLP-only --
the single most important addition for a vision-heavy venue like Pattern Recognition.

WHY THIS IS MOSTLY A ONE-LINE CHANGE
------------------------------------
extract.py's family_of() regexes already match ViT's HF parameter names:
    vit.encoder.layer.{i}.attention.attention.query.weight   -> attn_qkv
    vit.encoder.layer.{i}.attention.output.dense.weight      -> attn_out
    vit.encoder.layer.{i}.intermediate.dense.weight          -> ffn_intermediate
    vit.encoder.layer.{i}.output.dense.weight                -> ffn_output
    vit.pooler.dense.weight                                  -> pooler
So the ONLY required change to extract.py is registering a pinned revision:

    # in extract.py, add to REVISIONS:
    "google/vit-base-patch16-224": "<PIN_THIS_SHA>",   # fetch real SHA, see below

NOTE ViT position/patch embeddings are NOT 2-D Linear weights:
  * patch embedding is a Conv2d  -> 4-D tensor, skipped by extract()'s p.dim()==2 filter
  * position embedding is (1, num_patches+1, hidden) -> 3-D parameter, also skipped
That is fine: the FFN/attention families -- the paper's core comparison -- are what
we need for the aspect-ratio and bulk-conformity story. If you WANT the ViT position
embedding in Table 3, reshape it to 2-D explicitly (see vit_extra_embeddings() below).

ViT HAS NO MASKED-LM HEAD, so -- exactly like ALBERT -- it enters the Section 7
spectral characterization ONLY, never the Section 6 compression stress-test. Say
this in one sentence where ALBERT's exclusion is already explained.

USAGE
-----
1. Fetch and pin the real commit SHA (do NOT ship an unpinned model):
       python vit_support.py --print-sha google/vit-base-patch16-224
   then paste it into extract.py's REVISIONS.
2. Validate the family breakdown before integrating:
       python vit_support.py google/vit-base-patch16-224
3. Once REVISIONS is updated, run the normal pipeline unchanged:
       python run_characterize.py bert-base-uncased bert-large-uncased albert-base-v2 google/vit-base-patch16-224
"""
import sys, numpy as np
from collections import Counter

# Candidate vision Transformers (all use the same HF naming extract.py handles).
# Pin one (or several) after fetching the SHA with --print-sha.
VIT_REVISIONS = {
    "google/vit-base-patch16-224":  "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
    "google/vit-large-patch16-224": "<PIN_THIS_SHA>",   # optional: a scale pair, mirrors BERT-base/large
}


def print_sha(model_id):
    """Resolve the current main-branch commit SHA to pin for reproducibility."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(model_id)
    print(f'"{model_id}": "{info.sha}",')


def validate(model_id):
    """Extract and print the ViT family breakdown, so you can confirm the regexes
    catch the right matrices BEFORE wiring the revision into extract.py."""
    from transformers import AutoModel
    from extract import family_of, layer_index
    model = AutoModel.from_pretrained(model_id)      # unpinned OK for a dry-run only
    mats = []
    for name, p in model.named_parameters():
        if p.dim() != 2:
            continue
        fam = family_of(name)
        if fam is None:
            continue
        mats.append((name, fam, layer_index(name), tuple(p.shape)))
    c = Counter(f for _, f, _, _ in mats)
    print(f"\n{model_id}: {len(mats)} 2-D matrices matched")
    for fam, k in sorted(c.items()):
        ex = next(s for n, f, l, s in mats if f == fam)
        print(f"  {fam:18s} n={k:3d}  e.g. {ex}")
    unmatched = "  (if attention/FFN counts look wrong, ViT naming may need a "
    print(unmatched + "family_of() tweak -- but standard google/vit-* matches as-is)")


def vit_extra_embeddings(model):
    """OPTIONAL: expose ViT's position embedding as a 2-D matrix for Table 3.
    Returns list of extract()-style dicts. ViT pos emb has shape (1, T, H);
    we squeeze the batch axis to get a (T, H) operator analysable by the pipeline."""
    out = []
    for name, p in model.named_parameters():
        if "position_embeddings" in name and p.dim() == 3 and p.shape[0] == 1:
            W = p.detach().cpu().numpy().astype(np.float64)[0]   # (T, H)
            out.append(dict(name=name, family="emb_position", layer=-1,
                            shape=tuple(W.shape), W=W))
    return out


if __name__ == "__main__":
    if "--print-sha" in sys.argv:
        for mid in [a for a in sys.argv[1:] if not a.startswith("--")]:
            print_sha(mid)
    else:
        for mid in ([a for a in sys.argv[1:] if not a.startswith("--")]
                    or ["google/vit-base-patch16-224"]):
            validate(mid)
