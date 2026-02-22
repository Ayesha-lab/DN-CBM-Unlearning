"""
Main script for training binary blonde classification probe.
"""

import os
import torch
import argparse
import os.path as osp
from pathlib import Path

from .data_loader import get_concept_strengths_loader
from .model import create_model
from .trainer import BinaryProbeTrainer
from .mlflow_utils import maybe_start_mlflow_run
from .celeba_args import get_args


def main():
    args = get_args()

    print("\n" + "="*70)
    print("TRAIN BINARY CLASSIFICATION PROBE (BLONDE/NOT-BLONDE)")
    print("="*70)
    print(f"CelebA Root:         {args.celeba_root}")
    print(f"Train Activations:   {args.train_activations_path}")
    print(f"Val Activations:     {args.val_activations_path}")
    print(f"Test Activations:    {args.test_activations_path}")
    print(f"Attribute:           {args.attribute}")
    print(f"Num Epochs:          {args.num_epochs}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Learning Rate:       {args.learning_rate}")
    print(f"L1 Coefficient:      {args.l1_coeff}")
    print(f"Device:              {args.device}")
    print(f"MLflow Enabled:      {args.use_mlflow}")
    print("="*70 + "\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load training data
    print("→ Loading training data...")
    train_loader, n_train = get_concept_strengths_loader(
        activations_path=args.train_activations_path,
        celeba_root=args.celeba_root,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        attribute=args.attribute,
        device=args.device
    )

    # Load validation data (use CelebA val split)
    print("→ Loading validation data...")
    val_loader, n_val = get_concept_strengths_loader(
        activations_path=args.val_activations_path,
        celeba_root=args.celeba_root,
        split='val',
        batch_size=args.batch_size,
        shuffle=False,
        attribute=args.attribute,
        device=args.device
    )

    # Create model
    print("→ Creating model...")
    first_batch = next(iter(train_loader))
    n_concepts = first_batch[0].shape[1]
    print(f"  Number of concepts: {n_concepts}\n")

    model = create_model(n_concepts=n_concepts, device=args.device)

    # ---- MLflow: start run around training (ADD) ----
    run_name = args.mlflow_run_name
    if run_name is None:
        run_name = f"probe_{args.attribute}_l1_{args.l1_coeff}"

    mlflow_params = {
        **vars(args),
        "n_concepts": n_concepts,
        "n_train": n_train,
        "n_val": n_val,
    }
    mlflow_tags = {
        "task": "binary_linear_probe",
        "dataset": "celeba",
        "attribute": args.attribute,
    }

    with maybe_start_mlflow_run(
        enabled=args.use_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=run_name,
        tags=mlflow_tags,
        params=mlflow_params,
    ) as mlf:

        # Create trainer (pass mlf logger)
        trainer = BinaryProbeTrainer(
            model=model,
            device=args.device,
            learning_rate=args.learning_rate,
            l1_coeff=args.l1_coeff,
            mlflow_logger=mlf,          # <-- ADD
        )

        # Train
        print("→ Starting training...\n")
        best_acc = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            val_freq=args.val_freq
        )

        print("→ Loading test data...")
        test_loader, n_test = get_concept_strengths_loader(
            activations_path=args.test_activations_path,
            celeba_root=args.celeba_root,
            split='test',
            batch_size=args.batch_size,
            shuffle=False,
            attribute=args.attribute,
            device=args.device
        )

        test_metrics = trainer.evaluate(test_loader)
        print(
            f"→ Test Loss: {test_metrics['val/loss_total']:.6f} | "
            f"Test Acc: {test_metrics['val/acc_top1']:.4f}"
        )

        # If using MLflowLogger `mlf` in your with-block:
        mlf.log_metrics(
            {
                "test/loss_total": test_metrics["val/loss_total"],
                "test/loss_ce": test_metrics["val/loss_ce"],
                "test/acc_top1": test_metrics["val/acc_top1"],
            }
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

        # ---- MLflow: log artifacts (ADD) ----
        mlf.log_artifact(checkpoint_path, artifact_path="checkpoints")
        mlf.log_artifact(top_concepts_path, artifact_path="analysis")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()