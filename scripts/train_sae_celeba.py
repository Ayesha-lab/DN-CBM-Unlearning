"""
Train a Sparse Autoencoder (SAE) on CelebA CLIP features.

Mirrors scripts/train_sae.py but targets CelebA activations saved by
scripts/save_celeba_features.py, and uses MLflow instead of W&B for logging.

Usage:
    python scripts/train_sae_celeba.py \
        --sae_dataset celeba \
        --img_enc_name clip_RN50 \
        --expansion_factor 4 \
        --num_epochs 200 \
        --use_mlflow                      # optional
        --mlflow_tracking_uri file:./mlruns   # optional, default matches mlflow_setup.py
        --mlflow_experiment SAE_celeba        # optional
"""

import os
import sys
import datetime
from pathlib import Path

import torch
import numpy as np
import math
from time import time

from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    SparseAutoencoder,
)

from dncbm.custom_pipeline import Pipeline
from dncbm.arg_parser import get_common_parser
from dncbm.utils import common_init
import os.path as osp

# MLflow helpers (re-use existing project utilities)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from train_linear_probe_celeba.mlflow_utils import maybe_start_mlflow_run


# ──────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────
parser = get_common_parser()
parser.set_defaults(sae_dataset="celeba")

# MLflow flags (mirror the style used in train_linear_probe_celeba/celeba_args.py)
parser.add_argument("--use_mlflow", action="store_true", default=False,
                    help="Enable MLflow logging")
parser.add_argument("--mlflow_tracking_uri", type=str, default="file:./mlruns",
                    help='MLflow tracking URI, e.g. "file:./mlruns" or "http://host:5000"')
parser.add_argument("--mlflow_experiment", type=str, default="SAE_celeba",
                    help="MLflow experiment name (groups all SAE-celeba runs)")
parser.add_argument("--mlflow_run_name", type=str, default=None,
                    help="Override the auto-generated run name")

args = parser.parse_args()
common_init(args)
start_time = time()

# ──────────────────────────────────────────────────────────────────
# Unique run name
# common_init already builds args.config_name which encodes all the
# hyperparameters that distinguish one run from another:
#   lr{lr}_l1coeff{l1}_ef{expansion}_rf{resample}_hook{hook}_bs{bs}_epo{epo}
# We prefix it with dataset + encoder + date so it is globally unique
# across experiments and easy to read in the MLflow UI.
# ──────────────────────────────────────────────────────────────────
_date_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_auto_run_name = (
    f"SAE_{args.sae_dataset}"
    f"_{args.img_enc_name_for_saving}"
    f"_{args.hook_points[0]}"
    f"_{args.config_name}"   # already contains all key hyperparams
    f"_{_date_tag}"          # timestamp makes it unique even for identical hparams
)
mlflow_run_name = args.mlflow_run_name or _auto_run_name

# ──────────────────────────────────────────────────────────────────
# Vocabulary embeddings (for optional concept-name monitoring)
# ──────────────────────────────────────────────────────────────────
embeddings_path = osp.join(
    args.vocab_dir,
    f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth",
)
if not osp.exists(embeddings_path):
    raise FileNotFoundError(
        f"Vocabulary embeddings not found: {embeddings_path}\n"
        "Run the vocab embedding script first, or check args.vocab_dir."
    )
args.vocab_specific_embedding = torch.load(embeddings_path).to(args.device)

# ──────────────────────────────────────────────────────────────────
# Build the Sparse Autoencoder
# ──────────────────────────────────────────────────────────────────
autoencoder_input_dim: int = args.autoencoder_input_dim_dict[
    args.ae_input_dim_dict_key[args.modality]
]
n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
autoencoder = SparseAutoencoder(
    n_input_features=autoencoder_input_dim,
    n_learned_features=n_learned_features,
    n_components=len(args.hook_points),
).to(args.device)
print(f"Autoencoder created at {time() - start_time:.1f}s  "
      f"(input_dim={autoencoder_input_dim}, learned_features={n_learned_features})")

activations_dir = args.data_dir_activations[args.modality]
print(f"CelebA activations dir : {activations_dir}")
print(f"CLIP encoder           : {args.img_enc_name}")

# ──────────────────────────────────────────────────────────────────
# Loss, optimiser, resampler
# ──────────────────────────────────────────────────────────────────
loss = LossReducer(
    LearnedActivationsL1Loss(l1_coefficient=float(args.l1_coeff)),
    L2ReconstructionLoss(),
)

optimizer = AdamWithReset(
    params=autoencoder.parameters(),
    named_parameters=autoencoder.named_parameters(),
    lr=float(args.lr),
    betas=(float(args.adam_beta_1), float(args.adam_beta_2)),
    eps=float(args.adam_epsilon),
    weight_decay=float(args.adam_weight_decay),
    has_components_dim=True,
)

actual_resample_interval = 1
activation_resampler = ActivationResampler(
    resample_interval=actual_resample_interval,
    n_activations_activity_collate=actual_resample_interval,
    max_n_resamples=math.inf,
    n_learned_features=n_learned_features,
    resample_epoch_freq=args.resample_freq,
    resample_dataset_size=args.resample_dataset_size,
)
print(f"Loss / optimiser / resampler ready at {time() - start_time:.1f}s")

# ──────────────────────────────────────────────────────────────────
# Discover saved CelebA feature files
# ──────────────────────────────────────────────────────────────────
fnames = os.listdir(activations_dir)
train_fnames, train_val_fnames = [], []
for fname in fnames:
    full_path = osp.join(osp.abspath(activations_dir), fname)
    if fname.startswith("train_val"):
        train_val_fnames.append(full_path)
    elif fname.startswith("train"):
        train_fnames.append(full_path)

if args.val_freq == 0:
    train_fnames = train_fnames + train_val_fnames
    train_val_fnames = None

print(f"train files     : {train_fnames}")
print(f"train_val files : {train_val_fnames}")

# ──────────────────────────────────────────────────────────────────
# MLflow tags & params logged at run start
# ──────────────────────────────────────────────────────────────────
mlflow_tags = {
    "task":        "sae_training",
    "dataset":     args.sae_dataset,
    "img_encoder": args.img_enc_name,
    "hook_point":  args.hook_points[0],
}

mlflow_params = {
    "sae_dataset":        args.sae_dataset,
    "img_enc_name":       args.img_enc_name,
    "hook_point":         args.hook_points[0],
    "expansion_factor":   args.expansion_factor,
    "l1_coeff":           args.l1_coeff,
    "lr":                 args.lr,
    "num_epochs":         args.num_epochs,
    "train_sae_bs":       args.train_sae_bs,
    "resample_freq":      args.resample_freq,
    "resample_dataset_size": args.resample_dataset_size,
    "val_freq":           args.val_freq,
    "ckpt_freq":          args.ckpt_freq,
    "autoencoder_input_dim":  autoencoder_input_dim,
    "n_learned_features":     n_learned_features,
    "seed":               args.seed,
    "save_suffix":        args.save_suffix,
    "config_name":        args.config_name,
}

# ──────────────────────────────────────────────────────────────────
# Train inside the MLflow context
# ──────────────────────────────────────────────────────────────────
with maybe_start_mlflow_run(
        enabled=args.use_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=mlflow_run_name,
        tags=mlflow_tags,
        params=mlflow_params,
) as mlf:

    if args.use_mlflow:
        print(f"\n✓ MLflow run started")
        print(f"  Experiment : {args.mlflow_experiment}")
        print(f"  Run name   : {mlflow_run_name}")
        print(f"  Tracking   : {args.mlflow_tracking_uri}\n")

    # Pipeline (unchanged interface)
    pipeline = Pipeline(
        activation_resampler=activation_resampler,
        autoencoder=autoencoder,
        checkpoint_directory=Path(
            f"{args.save_dir_sae_ckpts[args.modality]}{args.save_suffix}"
        ),
        loss=loss,
        optimizer=optimizer,
        device=args.device,
        args=args,
    )
    print(f"Pipeline created at {time() - start_time:.1f}s — starting training...")

    pipeline.run_pipeline(
        train_batch_size=int(args.train_sae_bs),
        checkpoint_frequency=int(args.ckpt_freq),
        val_frequency=int(args.val_freq),
        num_epochs=args.num_epochs,
        train_fnames=train_fnames,
        train_val_fnames=train_val_fnames,
        start_time=start_time,
        resample_epoch_freq=args.resample_freq,
    )

    total_time = np.round(time() - start_time, 3)
    print(f"\n------- total time: {total_time}s -------")

    # Log final summary metrics to MLflow
    mlf.log_metric("total_training_time_s", total_time)
    mlf.log_metric("n_learned_features", n_learned_features)