"""
Assign semantic names to SAE concepts extracted from CelebA using CLIP vocabulary.
"""

import os
import torch
import argparse
import numpy as np
from pathlib import Path
import os.path as osp

from dncbm import arg_parser, utils, config
from sparse_autoencoder import SparseAutoencoder


def load_sae_and_concept_names(
    sae_checkpoint_path,
    vocab_embeddings_path,
    vocab_txt_path,
    autoencoder_input_dim,
    n_learned_features,
    device='cuda'
):
    """
    Load SAE checkpoint and compute concept name similarities.
    """
    print(f"→ Loading SAE checkpoint: {sae_checkpoint_path}")
    sae = SparseAutoencoder(
        n_input_features=autoencoder_input_dim,
        n_learned_features=n_learned_features,
        n_components=1
    ).to(device)
    
    state_dict = torch.load(sae_checkpoint_path, map_location=device)
    sae.load_state_dict(state_dict)
    sae.eval()
    print("✓ SAE loaded successfully")
    
    # Load vocabulary embeddings
    print(f"→ Loading vocabulary embeddings: {vocab_embeddings_path}")
    vocab_embeddings = torch.load(vocab_embeddings_path, map_location=device)
    vocab_embeddings = vocab_embeddings.float()
    print(f"✓ Vocab embeddings shape: {vocab_embeddings.shape}")
    
    # Load vocabulary text
    print(f"→ Loading vocabulary text: {vocab_txt_path}")
    with open(vocab_txt_path, 'r') as f:
        vocab_words = [line.strip() for line in f]
    print(f"✓ Loaded {len(vocab_words)} vocabulary words")
    
    # Get SAE decoder (which represents the learned concepts)
    # The decoder.weight contains the concept vectors
    concept_vectors = sae.decoder._weight.squeeze().detach().cpu()  # Shape: [n_learned_features, input_dim]
    print(f"→ Concept vectors shape: {concept_vectors.shape}")
    
    # Normalize concept vectors
    concept_vectors = (concept_vectors / (concept_vectors.norm(dim=-1, keepdim=True) + 1e-8)).to(device)
    
    # Normalize vocab embeddings
    vocab_embeddings = (vocab_embeddings / (vocab_embeddings.norm(dim=-1, keepdim=True) + 1e-8))
    
    # Compute similarity matrix: [num_concepts, num_vocab_words]
    print(f"→ Computing concept-vocabulary similarity matrix...")
    similarity_matrix = torch.mm(vocab_embeddings, concept_vectors)  # [n_vocab, n_concepts]
    print(f"✓ Similarity matrix shape [n_vocab, n_concepts]: {similarity_matrix.shape}")
    
    # Get top concept names
    top_indices = similarity_matrix.argmax(dim=0)  # [n_concepts]
    concept_names = [vocab_words[int(idx)] for idx in top_indices]
    similarities = similarity_matrix.max(dim=0).values  # [n_concepts] - use dim=0, not dim=1
    
    return concept_names, similarities.cpu().numpy(), similarity_matrix.cpu().numpy()


def main():
    """Assign names to CelebA concepts"""
    
    # Parse arguments
    parser = arg_parser.get_common_parser()
    parser.add_argument(
        "--sae_checkpoint_path",
        type=str,
        default="./checkpoints/clip_RN50_sparse_autoencoder_final.pt",
        help="Path to SAE checkpoint"
    )
    
    args = parser.parse_args()
    utils.common_init(args)
    
    print("\n" + "="*70)
    print("ASSIGN SEMANTIC NAMES TO CELEBA CONCEPTS")
    print("="*70)
    print(f"CLIP Model:              {args.img_enc_name}")
    print(f"Expansion Factor:        {args.expansion_factor}")
    print(f"SAE Checkpoint:          {args.sae_checkpoint_path}")
    print(f"Device:                  {args.device}")
    print("="*70 + "\n")
    
    # Check files exist
    print(f"→ Checking files...")
    if not os.path.exists(args.sae_checkpoint_path):
        print(f"✗ SAE checkpoint not found: {args.sae_checkpoint_path}")
        return False
    
    embeddings_path = osp.join(
        args.vocab_dir, 
        f"embeddings_{args.img_enc_name_for_saving}_{args.vocab_type}.pth"
    )
    vocab_txt_path = osp.join(
        args.vocab_dir, 
        f"{args.vocab_type}.txt"
    )
    
    if not os.path.exists(embeddings_path):
        print(f"✗ Embeddings file not found: {embeddings_path}")
        return False
    if not os.path.exists(vocab_txt_path):
        print(f"✗ Vocab text file not found: {vocab_txt_path}")
        return False
    
    print(f"✓ All files found\n")
    
    # Get SAE dimensions
    autoencoder_input_dim = args.autoencoder_input_dim_dict[
        args.ae_input_dim_dict_key[args.modality]
    ]
    n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
    
    print(f"→ SAE Configuration:")
    print(f"  Input dimension:       {autoencoder_input_dim}")
    print(f"  Learned features:      {n_learned_features}")
    print()
    
    # Load SAE and compute concept names
    try:
        concept_names, similarities, similarity_matrix = load_sae_and_concept_names(
            sae_checkpoint_path=args.sae_checkpoint_path,
            vocab_embeddings_path=embeddings_path,
            vocab_txt_path=vocab_txt_path,
            autoencoder_input_dim=autoencoder_input_dim,
            n_learned_features=n_learned_features,
            device=args.device
        )
    except Exception as e:
        print(f"✗ Error during concept naming: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Create output directory
    output_dir = args.save_dir["img"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Save concept names to CSV
    output_path = osp.join(output_dir, "concept_names_celeba.csv")
    
    print(f"→ Saving concept names to: {output_path}\n")
    print("Sample concept names (first 30):")
    print("-" * 80)
    print(f"{'Concept':>8} | {'Name':<30} | {'Similarity':>12}")
    print("-" * 80)
    
    with open(output_path, "w") as f:
        for idx in range(len(concept_names)):
            name = concept_names[idx]
            sim = similarities[idx]
            f.write(f"{idx},{name},{sim:.6f}\n")
            
            # Print first 30 for verification
            if idx < 30:
                print(f"{idx:8d} | {name:<30} | {sim:12.6f}")
    
    print("-" * 80)
    print(f"\n✓ Successfully assigned names to {len(concept_names)} concepts!")
    print(f"✓ Saved to: {output_path}")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)