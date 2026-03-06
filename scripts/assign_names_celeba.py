"""
Assign semantic concept names to a CelebA-trained SAE.

Usage:
    python scripts/assign_names_celeba.py \
        --sae_dataset celeba \
        --img_enc_name clip_RN50 \
        --expansion_factor 4
"""

import os
import os.path as osp
import torch
import numpy as np

from dncbm import arg_parser, utils, config
from sparse_autoencoder import SparseAutoencoder


def assign_celeba_concept_names(args):
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    embeddings_path = osp.join(
        args.vocab_dir,
        f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth",
    )
    vocab_txt_path = osp.join(args.vocab_dir, "clipdissect_20k.txt")

    for p in [embeddings_path, vocab_txt_path]:
        if not osp.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # ------------------------------------------------------------------
    # Load vocabulary
    # ------------------------------------------------------------------
    print(f"Loading vocabulary embeddings: {embeddings_path}")
    vocab_embeddings = torch.load(embeddings_path, map_location=args.device).float()

    print(f"Loading vocabulary text: {vocab_txt_path}")
    with open(vocab_txt_path, "r") as f:
        vocab_words = [line.strip() for line in f]
    print(f"  {len(vocab_words)} vocabulary words, "
          f"embeddings shape: {vocab_embeddings.shape}")

    # ------------------------------------------------------------------
    # Load SAE from checkpoint
    # ------------------------------------------------------------------
    autoencoder_input_dim: int = args.autoencoder_input_dim_dict[
        args.ae_input_dim_dict_key[args.modality]
    ]
    n_learned_features = int(autoencoder_input_dim * args.expansion_factor)

    autoencoder = SparseAutoencoder(
        n_input_features=autoencoder_input_dim,
        n_learned_features=n_learned_features,
        n_components=len(args.hook_points),
    ).to(args.device)

    autoencoder = utils.get_sae_ckpt(args, autoencoder)
    autoencoder.eval()
    print(f"SAE loaded  (input_dim={autoencoder_input_dim}, "
          f"n_learned_features={n_learned_features})")

    # ------------------------------------------------------------------
    # Extract concept vectors from the SAE decoder weight.
    #
    # decoder._weight raw shape: [n_components, n_learned_features, input_dim]
    #                        e.g. [1, 4096, 1024]
    #
    # We want concept_vectors: [n_learned_features, input_dim]
    #                      e.g. [4096, 1024]
    # so that  vocab_embeddings @ concept_vectors.T  works:
    #           [20000, 1024] @ [1024, 4096] → [20000, 4096]  ✓
    # ------------------------------------------------------------------
    raw_weight = autoencoder.decoder._weight.detach()
    print(f"Raw decoder weight shape: {raw_weight.shape}")

    # Flatten any leading component dimensions, then take [n_learned_features, input_dim]
    concept_vectors = raw_weight.reshape(-1, autoencoder_input_dim)
    # concept_vectors is now guaranteed [n_learned_features, input_dim]
    print(f"Concept vectors shape (after reshape): {concept_vectors.shape}")

    # L2-normalise both sides for cosine similarity
    concept_vectors = concept_vectors / (
            concept_vectors.norm(dim=-1, keepdim=True) + 1e-8
    )  # [n_concepts, input_dim]

    vocab_embeddings = vocab_embeddings / (
            vocab_embeddings.norm(dim=-1, keepdim=True) + 1e-8
    )  # [n_vocab, input_dim]

    # ------------------------------------------------------------------
    # similarity: [n_vocab, n_concepts]  →  argmax over vocab axis per concept
    # ------------------------------------------------------------------
    print("Computing concept–vocabulary similarity matrix...")
    # vocab_embeddings: [20000, 1024]
    # concept_vectors.T: [1024, 4096]
    # result:            [20000, 4096]
    similarity_matrix = torch.mm(vocab_embeddings, concept_vectors.T)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    top_concept_idxs = similarity_matrix.argmax(dim=0).cpu()  # [n_concepts]
    top_similarities = similarity_matrix.max(dim=0).values.cpu()  # [n_concepts]

    # ------------------------------------------------------------------
    # Write concept_names.csv
    # ------------------------------------------------------------------
    save_dir = args.save_dir["img"]
    os.makedirs(save_dir, exist_ok=True)
    out_path = osp.join(save_dir, "concept_names.csv")

    print(f"\nSaving concept names to: {out_path}")
    print(f"\n{'Concept':>8} | {'Name':<30} | {'Similarity':>12}")
    print("-" * 56)

    with open(out_path, "w") as f:
        for idx in range(top_concept_idxs.shape[0]):
            name = vocab_words[top_concept_idxs[idx].item()]
            sim  = top_similarities[idx].item()
            f.write(f"{idx},{name},{sim:.6f}\n")
            if idx < 30:
                print(f"{idx:8d} | {name:<30} | {sim:12.6f}")

    print("-" * 56)
    print(f"\n✓ Assigned names to {top_concept_idxs.shape[0]} concepts")
    print(f"✓ Saved to: {out_path}")


if __name__ == "__main__":
    parser = arg_parser.get_common_parser()
    parser.set_defaults(sae_dataset="celeba")
    args = parser.parse_args()
    utils.common_init(args)

    assign_celeba_concept_names(args)