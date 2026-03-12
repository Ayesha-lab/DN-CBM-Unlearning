"""
Concept-targeted unlearning for the CelebA binary probe (DN-CBM).

Pipeline:
  1. Load SAE activations (train split)
  2. Load CelebA attribute labels (train split)
  3. Identify top concept from all_concepts_ranked.csv
  4. Collect forget set  = top-K highest-activating images on that concept
     Collect retain set  = everything else
  5. Run CertifiedUnlearning (Phase 1: noisy SGD on retain set)
  6. Run post_unlearn_finetune (Phase 2: clean SGD on retain set)
  7. Compare probe weights before vs after — focus on top concept weight

All configs and metrics are logged to MLflow under experiment:
  "dncbm_unlearning_celeba"
"""

import os
import csv
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Reuse existing modules
from train_linear_probe_celeba.model import BinaryLinearProbe
from train_linear_probe_celeba.data_loader import load_celeba_labels
from train_linear_probe_celeba.mlflow_utils import maybe_start_mlflow_run
from Unlearning.certified_unlearning import (
    compute_noise_variance_with_regularization,
    clip_model_parameters,
    add_regularization_to_gradients,
    clip_gradients,
    add_noise_to_parameters,
    CertifiedUnlearning,
)

# ── CONFIG — all tunable parameters are here ──────────────────────────────────
CFG = {
    # Paths
    "probe_path"       : "./train_linear_probe_celeba/outputs/Pale_Skin/binary_probe.pt",
    "all_concepts_csv" : "./train_linear_probe_celeba/outputs/Pale_Skin/all_concepts_ranked.csv",
    "train_acts_path"  : "./data/activations_img/celeba/clip_RN50/out/train/sae_activations.pth",
    "celeba_root"      : "./data/celeba",
    "output_dir"       : "./unlearning/outputs/Pale_Skin",

    # Dataset
    "attribute"        : "Pale_Skin",
    "forget_k"         : 100,      # top-K images to forget
    "batch_size"       : 256,
    "device"           : "cuda" if torch.cuda.is_available() else "cpu",

    # Which concept to target:
    #   "positive" → top concept that pushes probe TOWARDS attribute (most common choice)
    #   "negative" → top concept that pushes probe AWAY from attribute
    #   "absolute" → highest |weight| regardless of sign
    "concept_polarity" : "positive",

    # Certified unlearning — Phase 1
    "epsilon"          : 10.0,
    "delta"            : 1e-5,
    "num_iterations"   : 10,
    "unlearn_lr"       : 0.001,    # γ
    "lambda_reg"       : 600.0,    # λ  (γλ = 0.6 ✓)
    "clip_norm_0"      : 250.0,    # C₀ — must be ≥ model L2 norm (200.3), use 250 for headroom
    "clip_norm_1"      : 1.0,      # C₁ — gradient clipping

    # Phase 2 fine-tuning
    "finetune_epochs"  : 10,
    "finetune_lr"      : 1e-3,
    "finetune_wd"      : 5e-4,
    "finetune_optimizer": "adam",  # "adam" or "sgd"

    # MLflow
    "use_mlflow"          : True,
    "mlflow_tracking_uri" : "file:./mlruns",
    "mlflow_experiment"   : "dncbm_unlearning_celeba",
    "mlflow_run_name"     : None,   # auto-generated if None
}
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_top_concept_idx(csv_path: str, polarity: str = "positive") -> int:
    """
    Load the top concept index from all_concepts_ranked.csv.

    Args:
        csv_path:  path to all_concepts_ranked.csv
                   columns: rank, concept_idx, name, weight
        polarity:  "positive"  → highest positive weight (concept that pushes
                                 the probe TOWARDS the attribute)
                   "negative"  → most negative weight (concept that pushes
                                 the probe AWAY from the attribute)
                   "absolute"  → highest |weight| regardless of sign
    Returns:
        concept_idx (int)
    """
    if polarity not in ("positive", "negative", "absolute"):
        raise ValueError(
            f"polarity must be 'positive', 'negative', or 'absolute', got {polarity!r}"
        )

    best_idx    = None
    best_weight = None

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)   # header: rank, concept_idx, name, weight
        for row in reader:
            w   = float(row["weight"])
            idx = int(row["concept_idx"])

            if polarity == "absolute":
                score = abs(w)
            elif polarity == "positive":
                if w <= 0:
                    continue
                score = w
            else:                     # "negative"
                if w >= 0:
                    continue
                score = -w            # most-negative → highest score

            if best_weight is None or score > best_weight:
                best_weight = score
                best_idx    = idx

    if best_idx is None:
        raise ValueError(
            f"No concept with polarity='{polarity}' found in {csv_path}. "
            "Check that the CSV contains weights of the expected sign."
        )
    return best_idx


def load_probe(probe_path: str, device: str):
    ckpt = torch.load(probe_path, map_location="cpu")
    n_concepts = ckpt["n_concepts"]
    model = BinaryLinearProbe(n_concepts=n_concepts)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"✓ Loaded probe: n_concepts={n_concepts}, "
          f"best_val_acc={ckpt.get('best_val_acc', float('nan')):.4f}")
    return model, n_concepts, ckpt


def build_forget_retain_loaders(acts, labels, top_concept_idx, forget_k, batch_size):
    concept_scores = acts[:, top_concept_idx]
    sorted_idx     = torch.argsort(concept_scores, descending=True)
    forget_indices = sorted_idx[:forget_k]
    retain_indices = sorted_idx[forget_k:]

    forget_loader = DataLoader(
        TensorDataset(acts[forget_indices], labels[forget_indices]),
        batch_size=batch_size, shuffle=False
    )
    retain_loader = DataLoader(
        TensorDataset(acts[retain_indices], labels[retain_indices]),
        batch_size=batch_size, shuffle=True
    )

    forget_mean = concept_scores[forget_indices].mean().item()
    retain_mean = concept_scores[retain_indices].mean().item()
    forget_min  = concept_scores[forget_indices].min().item()

    print(f"✓ Forget set : {len(forget_indices)} images | "
          f"mean act={forget_mean:.4f} | min act={forget_min:.4f}")
    print(f"✓ Retain set : {len(retain_indices)} images | mean act={retain_mean:.4f}")

    return forget_loader, retain_loader, {
        "forget_mean_act" : forget_mean,
        "forget_min_act"  : forget_min,
        "retain_mean_act" : retain_mean,
        "forget_n"        : len(forget_indices),
        "retain_n"        : len(retain_indices),
    }


def evaluate_probe(model, loader, device, split_name="eval"):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.float().unsqueeze(1).to(device)
            logits  = model(batch_X)
            total_loss += criterion(logits, batch_y).item()
            n_batches  += 1
            all_preds.append((logits > 0).long().squeeze(1).cpu())
            all_labels.append(batch_y.long().squeeze(1).cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc        = (all_preds == all_labels).float().mean().item()
    pos_mask   = all_labels == 1
    neg_mask   = all_labels == 0
    tpr     = (all_preds[pos_mask] == 1).float().mean().item() if pos_mask.any() else 0.0
    tnr     = (all_preds[neg_mask] == 0).float().mean().item() if neg_mask.any() else 0.0
    bal_acc = (tpr + tnr) / 2.0
    avg_loss = total_loss / max(1, n_batches)

    metrics = {"acc": acc, "bal_acc": bal_acc, "tpr": tpr, "tnr": tnr, "loss": avg_loss}
    print(f"  [{split_name}] loss={avg_loss:.4f} | acc={acc:.4f} | "
          f"bal_acc={bal_acc:.4f} | TPR={tpr:.4f} | TNR={tnr:.4f}")
    return metrics


def print_weight_diff(weights_before, weights_after, top_concept_idx, top_k=10):
    diff     = weights_after - weights_before
    abs_diff = diff.abs()

    w_before = weights_before[top_concept_idx].item()
    w_after  = weights_after[top_concept_idx].item()
    delta    = diff[top_concept_idx].item()
    pct      = 100 * delta / (abs(w_before) + 1e-9)

    print(f"\n  Top-concept (idx={top_concept_idx}):")
    print(f"    weight before : {w_before:+.6f}")
    print(f"    weight after  : {w_after:+.6f}")
    print(f"    Δ weight      : {delta:+.6f}  ({pct:.1f}%)")

    top_changed = torch.argsort(abs_diff, descending=True)[:top_k]
    print(f"\n  Top-{top_k} most changed weights:")
    print(f"  {'rank':>4}  {'concept':>8}  {'before':>12}  {'after':>12}  {'delta':>12}")
    for rank, idx in enumerate(top_changed):
        i = int(idx)
        print(f"  {rank+1:>4}  {i:>8}  "
              f"{weights_before[i]:>+12.6f}  "
              f"{weights_after[i]:>+12.6f}  "
              f"{diff[i]:>+12.6f}")

    return {
        "top_concept_weight_before"     : w_before,
        "top_concept_weight_after"      : w_after,
        "top_concept_weight_delta"      : delta,
        "top_concept_weight_pct_change" : pct,
    }


# ── Patched loss: BCEWithLogitsLoss for [B,1] probe output ───────────────────

class FlatBCELoss(nn.Module):
    """BCEWithLogitsLoss that handles [B,1] logits and Long targets."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        return self.bce(logits, target.float().unsqueeze(1))


# ── CelebA-aware subclass of CertifiedUnlearning ─────────────────────────────

class CelebACertifiedUnlearning(CertifiedUnlearning):
    """
    Subclass of CertifiedUnlearning that:
      - Removes the MNIST view() reshape (concept vectors are already flat)
      - Uses BCEWithLogitsLoss instead of CrossEntropyLoss
      - Accepts an MLflowLogger for metric logging
    """

    def __init__(self, model, device, mlf):
        super().__init__(model=model, device=device)
        self.mlf = mlf

    def unlearn(self, retain_loader, epsilon, delta, num_iterations,
                learning_rate, lambda_reg, clip_norm_0, clip_norm_1,
                loss_fn=None, verbose=True):

        if loss_fn is None:
            loss_fn = FlatBCELoss()

        sigma_squared = compute_noise_variance_with_regularization(
            epsilon, delta, num_iterations, learning_rate, lambda_reg,
            clip_norm_0, clip_norm_1
        )

        if verbose:
            print(f"\n{'='*70}")
            print("PHASE 1: NOISY UNLEARNING (CelebA binary probe)")
            print(f"{'='*70}")
            print(f"  ε={epsilon}, δ={delta}, T={num_iterations}")
            print(f"  γ={learning_rate}, λ={lambda_reg}, γλ={learning_rate*lambda_reg:.3f}")
            print(f"  C₀={clip_norm_0}, C₁={clip_norm_1}")
            print(f"  σ²={sigma_squared:.4e},  σ={np.sqrt(sigma_squared):.4e}\n")

        self.mlf.log_metric("unlearn/sigma_squared", sigma_squared)
        self.mlf.log_metric("unlearn/sigma", float(np.sqrt(sigma_squared)))

        clip_model_parameters(self.model, clip_norm_0)

        import torch.optim as optim
        from tqdm import tqdm
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=0.0)
        self.model.train()

        for iteration in tqdm(range(num_iterations),
                              desc="Phase 1: Noisy Unlearning", disable=not verbose):
            iter_loss, n_batches = 0.0, 0
            for data, target in retain_loader:
                data   = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss   = loss_fn(output, target)
                loss.backward()
                add_regularization_to_gradients(self.model, lambda_reg)
                clip_gradients(self.model, clip_norm_1)
                optimizer.step()
                add_noise_to_parameters(self.model, sigma_squared, self.device)
                iter_loss += loss.item()
                n_batches += 1

            avg_loss = iter_loss / max(1, n_batches)
            self.mlf.log_metric("unlearn/phase1_loss", avg_loss, step=iteration)

            if verbose:
                print(f"  iter {iteration+1}/{num_iterations} | avg_loss={avg_loss:.4f}")

        if verbose:
            print("✓ Phase 1 complete.\n")
        return self.model

    def post_unlearn_finetune(self, retain_loader, num_epochs, learning_rate,
                              loss_fn=None, weight_decay=0.0, verbose=True,
                              optimizer_type="adam"):

        if loss_fn is None:
            loss_fn = FlatBCELoss()

        if verbose:
            print(f"\n{'='*70}")
            print(f"PHASE 2: FINE-TUNING  (optimizer={optimizer_type}, "
                  f"lr={learning_rate}, wd={weight_decay})")
            print(f"{'='*70}\n")

        import torch.optim as optim
        from tqdm import tqdm

        if optimizer_type == "adam":
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay)

        self.model.train()

        for epoch in tqdm(range(num_epochs),
                          desc="Phase 2: Fine-Tuning", disable=not verbose):
            epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
            for data, target in retain_loader:
                data   = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss   = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss    += loss.item()
                preds          = (output > 0).long().squeeze(1)
                epoch_correct += (preds == target).sum().item()
                epoch_total   += target.size(0)

            avg_loss = epoch_loss / max(1, len(retain_loader))
            avg_acc  = epoch_correct / max(1, epoch_total)
            self.mlf.log_metric("unlearn/phase2_loss", avg_loss, step=epoch)
            self.mlf.log_metric("unlearn/phase2_acc",  avg_acc,  step=epoch)

            if verbose:
                print(f"  epoch {epoch+1}/{num_epochs} | "
                      f"loss={avg_loss:.4f} | acc={avg_acc:.4f}")

        if verbose:
            print("\n✓ Phase 2 complete.\n")
        return self.model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    C = CFG
    os.makedirs(C["output_dir"], exist_ok=True)

    run_name = C["mlflow_run_name"] or (
        f"unlearn_{C['attribute']}_k{C['forget_k']}"
        f"_eps{C['epsilon']}_T{C['num_iterations']}"
        f"_pol{C['concept_polarity']}"        # ← polarity in run name so runs are distinguishable
    )

    # Everything in CFG gets logged — including concept_polarity and finetune_optimizer
    mlflow_params = {k: v for k, v in C.items()
                     if not k.startswith("mlflow") and k != "use_mlflow"}

    mlflow_tags = {
        "task"      : "concept_unlearning",
        "dataset"   : "celeba",
        "attribute" : C["attribute"],
        "model"     : "BinaryLinearProbe",
        "method"    : "CertifiedUnlearning",
        "polarity"  : C["concept_polarity"],
    }

    with maybe_start_mlflow_run(
            enabled         = C["use_mlflow"],
            tracking_uri    = C["mlflow_tracking_uri"],
            experiment_name = C["mlflow_experiment"],
            run_name        = run_name,
            tags            = mlflow_tags,
            params          = mlflow_params,   # ← concept_polarity + finetune_optimizer logged here
    ) as mlf:

        print("=" * 70)
        print("DN-CBM CONCEPT UNLEARNING — CelebA Binary Probe")
        print("=" * 70)
        print(f"  Experiment       : {C['mlflow_experiment']}")
        print(f"  Run name         : {run_name}")
        print(f"  Attribute        : {C['attribute']}")
        print(f"  Forget K         : {C['forget_k']}")
        print(f"  Concept polarity : {C['concept_polarity']}")
        print(f"  Finetune optim   : {C['finetune_optimizer']}")
        print(f"  Device           : {C['device']}\n")

        # ── 0. Sanity check noise variance ──────────────────────────────────
        sigma_sq = compute_noise_variance_with_regularization(
            C["epsilon"], C["delta"], C["num_iterations"],
            C["unlearn_lr"], C["lambda_reg"], C["clip_norm_0"], C["clip_norm_1"]
        )
        print(f"  σ² = {sigma_sq:.4e}  |  σ = {np.sqrt(sigma_sq):.4e}")
        if sigma_sq > 1.0:
            print(f"  ⚠  WARNING: σ²={sigma_sq:.2e} is large — weights may be destroyed.")
            print(f"     Increase ε or num_iterations to reduce noise.\n")

        # ── 1. Load top concept  (polarity comes from CFG) ───────────────────
        top_concept_idx = load_top_concept_idx(
            C["all_concepts_csv"],
            polarity=C["concept_polarity"],   # ← uses CFG value, not hardcoded default
        )
        print(f"\n✓ Top concept index: {top_concept_idx}  (polarity='{C['concept_polarity']}')")
        # concept_polarity and finetune_optimizer are already in mlflow_params above;
        # log top_concept_idx separately since it's derived, not configured
        mlf.log_params({"top_concept_idx": top_concept_idx})

        # ── 2. Load probe ────────────────────────────────────────────────────
        model, n_concepts, ckpt = load_probe(C["probe_path"], C["device"])
        weights_before = model.linear.weight.detach().cpu().squeeze().clone()

        with torch.no_grad():
            model_l2 = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        print(f"  Model L2 norm: {model_l2:.4f}  "
              f"(C₀={C['clip_norm_0']} — {'✓ OK' if C['clip_norm_0'] >= model_l2 else '⚠ TOO SMALL'})")
        mlf.log_metric("model/l2_norm_before", model_l2)
        mlf.log_metric("model/top_concept_weight_before",
                       weights_before[top_concept_idx].item())

        # ── 3. Load SAE activations + labels ────────────────────────────────
        print(f"\n→ Loading SAE activations: {C['train_acts_path']}")
        acts = torch.load(C["train_acts_path"], map_location="cpu")
        if acts.ndim == 3 and acts.shape[1] == 1:
            acts = acts.squeeze(1)
        print(f"✓ Activations: {acts.shape}")

        print(f"→ Loading CelebA labels for '{C['attribute']}'")
        labels = load_celeba_labels(
            C["celeba_root"], split='train', attribute=C["attribute"]
        )
        print(f"✓ Labels: {labels.shape} | positives: {labels.sum().item()}")
        assert acts.shape[0] == labels.shape[0], (
            f"Mismatch: acts={acts.shape[0]}, labels={labels.shape[0]}"
        )

        # ── 4. Build forget / retain loaders ────────────────────────────────
        print(f"\n→ Building forget/retain split (k={C['forget_k']})...")
        forget_loader, retain_loader, split_stats = build_forget_retain_loaders(
            acts, labels, top_concept_idx, C["forget_k"], C["batch_size"]
        )
        mlf.log_params(split_stats)

        # ── 5. Baseline evaluation ───────────────────────────────────────────
        print("\n── Baseline (before unlearning) ────────────────────────────────")
        print("  Forget set:")
        mfb = evaluate_probe(model, forget_loader, C["device"], "forget")
        print("  Retain set:")
        mrb = evaluate_probe(model, retain_loader, C["device"], "retain")
        mlf.log_metrics({f"before/forget_{k}": v for k, v in mfb.items()})
        mlf.log_metrics({f"before/retain_{k}": v for k, v in mrb.items()})

        # ── 6. Unlearning ─────────────────────────────────────────────────────
        unlearner = CelebACertifiedUnlearning(
            model=model, device=C["device"], mlf=mlf
        )

        model = unlearner.unlearn(
            retain_loader  = retain_loader,
            epsilon        = C["epsilon"],
            delta          = C["delta"],
            num_iterations = C["num_iterations"],
            learning_rate  = C["unlearn_lr"],
            lambda_reg     = C["lambda_reg"],
            clip_norm_0    = C["clip_norm_0"],
            clip_norm_1    = C["clip_norm_1"],
            verbose        = True,
        )

        model = unlearner.post_unlearn_finetune(
            retain_loader  = retain_loader,
            num_epochs     = C["finetune_epochs"],
            learning_rate  = C["finetune_lr"],
            weight_decay   = C["finetune_wd"],
            optimizer_type = C["finetune_optimizer"],   # ← correctly wired from CFG
            verbose        = True,
        )

        # ── 7. Post-unlearning evaluation ────────────────────────────────────
        print("── After unlearning ────────────────────────────────────────────")
        print("  Forget set:")
        mfa = evaluate_probe(model, forget_loader, C["device"], "forget")
        print("  Retain set:")
        mra = evaluate_probe(model, retain_loader, C["device"], "retain")
        mlf.log_metrics({f"after/forget_{k}": v for k, v in mfa.items()})
        mlf.log_metrics({f"after/retain_{k}": v for k, v in mra.items()})

        # ── 8. Weight analysis ───────────────────────────────────────────────
        weights_after = model.linear.weight.detach().cpu().squeeze().clone()

        print("\n── Weight change analysis ──────────────────────────────────────")
        weight_metrics = print_weight_diff(weights_before, weights_after, top_concept_idx)
        mlf.log_metrics(weight_metrics)

        with torch.no_grad():
            model_l2_after = sum(p.norm().item() ** 2 for p in model.parameters()) ** 0.5
        mlf.log_metric("model/l2_norm_after", model_l2_after)
        mlf.log_metric("model/top_concept_weight_after",
                       weights_after[top_concept_idx].item())

        mlf.log_metrics({
            "delta/forget_acc"     : mfa["acc"]     - mfb["acc"],
            "delta/forget_bal_acc" : mfa["bal_acc"] - mfb["bal_acc"],
            "delta/retain_acc"     : mra["acc"]     - mrb["acc"],
            "delta/retain_bal_acc" : mra["bal_acc"] - mrb["bal_acc"],
        })

        # ── 9. Save checkpoint ───────────────────────────────────────────────
        out_ckpt = os.path.join(
            C["output_dir"],
            f"binary_probe_unlearned_k{C['forget_k']}_pol{C['concept_polarity']}.pt"
        )
        torch.save({
            "model_state"           : model.state_dict(),
            "n_concepts"            : n_concepts,
            "weights_before"        : weights_before,
            "weights_after"         : weights_after,
            "weight_delta"          : weights_after - weights_before,
            "top_concept_idx"       : top_concept_idx,
            "concept_polarity"      : C["concept_polarity"],
            "metrics_forget_before" : mfb,
            "metrics_forget_after"  : mfa,
            "metrics_retain_before" : mrb,
            "metrics_retain_after"  : mra,
            "config"                : C,
        }, out_ckpt)
        print(f"\n✓ Saved unlearned checkpoint → {out_ckpt}")

        mlf.log_artifact(out_ckpt, artifact_path="checkpoints")
        print("✓ Checkpoint logged to MLflow.")
        print("=" * 70)


if __name__ == "__main__":
    main()