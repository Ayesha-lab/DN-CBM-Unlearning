"""
Main script for training binary blonde classification probe.
"""

import os
import torch
import argparse
import os.path as osp
from pathlib import Path

from data_loader import get_concept_strengths_loader
from model import create_model
from trainer import BinaryProbeTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train binary classification probe on CelebA concepts"
    )
    
    # Data arguments
    parser.add_argument(
        "--celeba_root",
        type=str,
        default="./data/celeba",
        help="Path to CelebA root directory"
    )
    parser.add_argument(
        "--activations_path",
        type=str,
        default="./data/activations_img/celeba/clip_RN50/out/train/sae_activations.pth",
        help="Path to SAE activations file"
    )
    parser.add_argument(
        "--attribute",
        type=str,
        default="Blond_Hair",
        help="CelebA attribute to predict"
    )
    
    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--l1_coeff",
        type=float,
        default=0.0,
        help="L1 sparsity coefficient"
    )
    parser.add_argument(
        "--val_freq",
        type=int,
        default=5,
        help="Validation frequency"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./train_linear_probe_celeba/outputs",
        help="Directory to save outputs"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("TRAIN BINARY CLASSIFICATION PROBE (BLONDE/NOT-BLONDE)")
    print("="*70)
    print(f"CelebA Root:         {args.celeba_root}")
    print(f"Activations Path:    {args.activations_path}")
    print(f"Attribute:           {args.attribute}")
    print(f"Num Epochs:          {args.num_epochs}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Learning Rate:       {args.learning_rate}")
    print(f"L1 Coefficient:      {args.l1_coeff}")
    print(f"Device:              {args.device}")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training data
    print("→ Loading training data...")
    train_loader, n_samples = get_concept_strengths_loader(
        activations_path=args.activations_path,
        celeba_root=args.celeba_root,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        attribute=args.attribute,
        device=args.device
    )
    
    # Load validation data (use CelebA val split)
    print("→ Loading validation data...")
    val_loader, _ = get_concept_strengths_loader(
        activations_path=args.activations_path,  # TODO: use actual val activations
        celeba_root=args.celeba_root,
        split='val',
        batch_size=args.batch_size,
        shuffle=False,
        attribute=args.attribute,
        device=args.device
    )
    
    # Create model
    print("→ Creating model...")
    # Get number of concepts from first batch
    first_batch = next(iter(train_loader))
    n_concepts = first_batch[0].shape[1]
    print(f"  Number of concepts: {n_concepts}\n")
    
    model = create_model(n_concepts=n_concepts, device=args.device)
    
    # Create trainer
    trainer = BinaryProbeTrainer(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate,
        l1_coeff=args.l1_coeff
    )
    
    # Train
    print("→ Starting training...\n")
    best_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        val_freq=args.val_freq
    )
    
    # Save model
    print(f"\n→ Saving model...")
    checkpoint = {
        'model_state': model.state_dict(),
        'n_concepts': n_concepts,
        'best_val_acc': best_acc,
        'weights': trainer.get_model_weights(),
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'val_accs': trainer.val_accs
    }
    
    checkpoint_path = osp.join(args.output_dir, 'binary_probe_blonde.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Saved to: {checkpoint_path}\n")
    
    # Save top contributing concepts
    print("→ Top contributing concepts (highest absolute weights):")
    weights = trainer.get_model_weights()
    top_indices = torch.argsort(torch.abs(weights), descending=True)[:10]
    
    top_concepts_path = osp.join(args.output_dir, 'top_concepts.txt')
    with open(top_concepts_path, 'w') as f:
        for rank, idx in enumerate(top_indices):
            weight = weights[idx].item()
            f.write(f"{rank+1:2d}. Concept {idx:5d} | Weight: {weight:+.6f}\n")
            print(f"  {rank+1:2d}. Concept {idx:5d} | Weight: {weight:+.6f}")
    
    print(f"\n✓ Saved concept rankings to: {top_concepts_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()