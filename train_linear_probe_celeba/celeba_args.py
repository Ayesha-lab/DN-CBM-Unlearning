import argparse

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train binary classification probe on CelebA concepts"
    )

    # Data
    parser.add_argument("--celeba_root", type=str, default="./data/celeba")
    parser.add_argument("--train_activations_path", type=str,
        default="./data/activations_img/celeba/clip_RN50/out/train/sae_activations.pth")
    parser.add_argument("--val_activations_path", type=str,
        default="./data/activations_img/celeba/clip_RN50/out/val/sae_activations.pth")
    parser.add_argument("--test_activations_path", type=str,
        default="./data/activations_img/celeba/clip_RN50/out/test/sae_activations.pth")
    parser.add_argument("--attribute", type=str, default="Blond_Hair")

    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--l1_coeff", type=float, default=0.0)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # Output
    parser.add_argument("--output_dir", type=str, default="./train_linear_probe_celeba/outputs")
    parser.add_argument("--concept_names_path", type=str,                          # ← NEW
                        default="./checkpoints/clip_RN50_concept_name.csv",
                        help="Path to CSV with columns: concept_idx, name, similarity")

                    # MLflow
    parser.add_argument("--use_mlflow", action="store_true")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="file:./mlruns")
    parser.add_argument("--mlflow_experiment", type=str, default="linear_probe_celeba")
    parser.add_argument("--mlflow_run_name", type=str, default=None)

    return parser.parse_args()