"""
Main script for training binary blonde classification probe.
"""

import os
import csv
import torch
import os.path as osp
from pathlib import Path

from .data_loader import get_concept_strengths_loader
from .model import create_model
from .trainer import BinaryProbeTrainer
from .mlflow_utils import maybe_start_mlflow_run
from .celeba_args import get_args


# ── helpers ───────────────────────────────────────────────────────────────────

def load_concept_names(csv_path: str) -> dict:
    """
    Load concept names from a CSV file with columns: concept_idx, name, similarity.
    Returns a dict {concept_idx (int): name (str)}.
    Falls back gracefully if the file is missing.
    """
    names = {}
    if not osp.exists(csv_path):
        print(f"⚠  Concept names CSV not found: {csv_path}  (indices will be used instead)")
        return names
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    names[int(row[0])] = row[1].strip()
                except ValueError:
                    pass  # skip header rows or malformed lines
    print(f"✓ Loaded {len(names)} concept names from: {csv_path}")
    return names


def concept_label(idx: int, names: dict) -> str:
    """Return 'name (idx)' if name is available, else 'Concept idx'."""
    if idx in names:
        return f"{names[idx]} ({idx})"
    return f"Concept {idx}"


# ── main ──────────────────────────────────────────────────────────────────────

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
    print(f"Concept Names CSV:   {args.concept_names_path}")
    print("="*70 + "\n")

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.attribute)
    os.makedirs(output_dir, exist_ok=True)

    # Load concept names lookup
    concept_names = load_concept_names(args.concept_names_path)

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

    # Load validation data
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

        trainer = BinaryProbeTrainer(
            model=model,
            device=args.device,
            learning_rate=args.learning_rate,
            l1_coeff=args.l1_coeff,
            mlflow_logger=mlf,
        )

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
            f"Test Acc: {test_metrics['val/acc']:.4f} | "
            f"Test BalAcc: {test_metrics['val/bal_acc']:.4f} | "
            f"Test TPR: {test_metrics['val/tpr']:.4f} | "
            f"Test TNR: {test_metrics['val/tnr']:.4f}"
        )

        mlf.log_metrics(
            {
                "test/loss_total": test_metrics["val/loss_total"],
                "test/loss_ce":    test_metrics["val/loss_ce"],
                "test/acc":        test_metrics["val/acc"],
                "test/bal_acc":    test_metrics["val/bal_acc"],
                "test/tpr":        test_metrics["val/tpr"],
                "test/tnr":        test_metrics["val/tnr"],
            }
        )

        # Save model checkpoint
        print(f"\n→ Saving model...")
        weights = trainer.get_model_weights()   # shape [n_concepts]
        checkpoint = {
            'model_state':  model.state_dict(),
            'n_concepts':   n_concepts,
            'best_val_acc': best_acc,
            'weights':      weights,
            'train_losses': trainer.train_losses,
            'val_losses':   trainer.val_losses,
            'val_accs':     trainer.val_accs,
        }
        checkpoint_path = osp.join(output_dir, 'binary_probe_blonde.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved to: {checkpoint_path}\n")

        # ── Top-10 named concepts (printed + saved) ───────────────────────
        print("→ Top contributing concepts (highest absolute weights):")
        top_indices = torch.argsort(torch.abs(weights), descending=True)[:10]

        top_concepts_path = osp.join(output_dir, 'top_concepts.txt')
        with open(top_concepts_path, 'w') as f:
            for rank, idx in enumerate(top_indices):
                w      = weights[idx].item()
                label  = concept_label(int(idx), concept_names)
                line   = f"  {rank+1:2d}. {label:<35} | Weight: {w:+.6f}"
                f.write(line + "\n")
                print(line)

        print(f"\n✓ Saved top-10 to: {top_concepts_path}")

        # ── Full ranked list: ALL concepts with signed weight + name ──────
        all_ranked_path = osp.join(output_dir, 'all_concepts_ranked.csv')
        sorted_indices  = torch.argsort(torch.abs(weights), descending=True)

        with open(all_ranked_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "concept_idx", "name", "weight"])
            for rank, idx in enumerate(sorted_indices):
                idx_int = int(idx)
                w       = weights[idx].item()
                name    = concept_names.get(idx_int, "")
                writer.writerow([rank + 1, idx_int, name, f"{w:+.6f}"])

        print(f"✓ Saved all {n_concepts} concepts (ranked) to: {all_ranked_path}")

        # MLflow artifacts
        mlf.log_artifact(checkpoint_path,   artifact_path="checkpoints")
        mlf.log_artifact(top_concepts_path, artifact_path="../analysis")
        mlf.log_artifact(all_ranked_path, artifact_path="../analysis")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()