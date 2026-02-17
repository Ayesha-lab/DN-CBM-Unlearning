"""
Certified Unlearning for Neural Networks
Based on Koloskova et al., ICML 2025

This code extracts the core unlearning logic from:
https://github.com/stair-lab/certified-unlearning-neural-networks-icml-2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional
from Models.mlp import MLP
from tqdm import tqdm
import mlflow
import itertools


def compute_noise_variance_with_regularization(
        epsilon: float,
        delta: float,
        num_iterations: int,
        learning_rate: float,
        lambda_reg: float,
        clip_norm_0: float,
        clip_norm_1: float
) -> float:
    """
    Compute required noise variance for (ε, δ)-unlearning with regularization. (Theorem 4.1, Case 2)
    Args:
        epsilon: Privacy parameter ε (should satisfy ε ≤ 3log(1/δ))
        delta: Privacy parameter δ ∈ (0, 1)
        num_iterations: Number of unlearning iterations T
        learning_rate: Learning rate γ
        lambda_reg: L2 regularization coefficient λ > 0
        clip_norm_0: Initial model clipping threshold C₀
        clip_norm_1: Gradient clipping threshold C₁

    Returns:
        Required noise variance σ²

    Raises:
        ValueError: If constraints are violated
    """
    T = num_iterations
    gamma = learning_rate
    lam = lambda_reg

    # Validate constraint: γλ ∈ (1/2, 1)
    gamma_lambda = gamma * lam
    # if not (0.5 < gamma_lambda < 1.0):
    #     raise ValueError(
    #         f"Constraint violated: γλ = {gamma_lambda:.4f} must be in (0.5, 1.0). "
    #         f"Adjust learning_rate={gamma} or lambda_reg={lam}"
    #     )

    # Validate privacy parameter
    if epsilon > 3 * np.log(1 / delta):
        raise ValueError(
            f"ε = {epsilon} exceeds 3·log(1/δ) = {3 * np.log(1 / delta):.4f}. "
            f"Decrease ε or increase δ."
        )

    # σ² = (72γλ log(1/δ)) / ε² * (C₀(1-γλ)^T + C₁/λ)²
    numerator = 72 * gamma * lam * np.log(1 / delta)
    denominator = epsilon ** 2

    # Compute clip term
    decay_factor = (1 - gamma_lambda) ** T
    clip_term = (clip_norm_0 * decay_factor + clip_norm_1 / lam) ** 2

    sigma_squared = (numerator / denominator) * clip_term

    return sigma_squared


def clip_model_parameters(
        model: nn.Module,
        clip_norm: float
) -> None:
    """
    Clip model parameters by their L2 norm (initial clipping C₀).

    Implements: Π_{C}(x) := x · min{C / ||x||, 1}

    Args:
        model: Model whose parameters to clip
        clip_norm: Clipping threshold C
    """
    with torch.no_grad():
        for param in model.parameters():
            param_norm = torch.norm(param)
            if param_norm > clip_norm:
                param.data *= (clip_norm / param_norm)


def add_regularization_to_gradients(
        model: nn.Module,
        lambda_reg: float
) -> None:
    """
    Add L2 regularization term λx to gradients.

    This must happen BEFORE gradient clipping (not via weight_decay).
    Implements: grad := grad + λ·param

    Args:
        model: Model whose parameters to regularize
        lambda_reg: Regularization coefficient λ
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.grad.add_(param.data, alpha=lambda_reg)


def clip_gradients(
        model: nn.Module,
        clip_norm: float
) -> None:
    """
    Clip gradients by their L2 norm (Π_{C₁}).

    Implements: Π_{C}(g) := g · min{C / ||g||, 1}

    Args:
        model: Model whose gradients to clip
        clip_norm: Clipping threshold C₁
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)


def add_noise_to_parameters(
        model: nn.Module,
        noise_variance: float,
        device: torch.device
) -> None:
    """
    Add Gaussian noise to model parameters AFTER optimizer step.

    Implements: param := param + ξ, where ξ ~ N(0, σ²I)

    Args:
        model: Model whose parameters to perturb
        noise_variance: Variance σ²
        device: Device for noise tensor
    """
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param, device=device) * np.sqrt(noise_variance)
            param.data.add_(noise)


class CertifiedUnlearning:
    """
    Certified unlearning via noisy gradient descent with regularization.

    Implements the gradient clipping variant (Equation 3) from the paper
    with L2 regularization (λ > 0).

    The unlearning process has two phases:
    1. Noisy fine-tuning: T iterations with gradient clipping + noise
    2. Standard fine-tuning: Recover accuracy without noise/clipping
    """

    def __init__(
            self,
            model: nn.Module,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize certified unlearning.

        Args:
            model: Pre-trained PyTorch model to unlearn
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()  # Start in eval mode before unlearning

    def unlearn(
            self,
            retain_loader: DataLoader,
            epsilon: float,
            delta: float,
            num_iterations: int,
            learning_rate: float,
            lambda_reg: float,
            clip_norm_0: float,
            clip_norm_1: float,
            loss_fn: Optional[nn.Module] = None,
            verbose: bool = True
    ) -> nn.Module:
        """
        Phase 1: Noisy fine-tuning with gradient clipping and regularization.

        Performs T iterations of:
            1. Compute loss on retain set
            2. Backward pass
            3. Add L2 regularization term: grad += λ·param
            4. Clip gradients: Π_{C₁}(grad)
            5. Optimizer step
            6. Add noise to parameters: param += ξ ~ N(0, σ²I)

        Args:
            retain_loader: DataLoader for data to retain (D_r)
            epsilon: Privacy parameter ε
            delta: Privacy parameter δ
            num_iterations: Number of unlearning iterations T
            learning_rate: Learning rate γ
            lambda_reg: Regularization coefficient λ > 0
            clip_norm_0: Initial model clipping threshold C₀
            clip_norm_1: Gradient clipping threshold C₁
            loss_fn: Loss function (default: CrossEntropyLoss)
            verbose: Print progress and validation info

        Returns:
            Model after noisy unlearning phase

        Raises:
            ValueError: If constraints are violated
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        # Compute required noise variance (validates constraints internally)
        sigma_squared = compute_noise_variance_with_regularization(
            epsilon, delta, num_iterations, learning_rate,
            lambda_reg, clip_norm_0, clip_norm_1
        )

        if verbose:
            print("\n" + "="*70, flush=True)
            print("PHASE 1: NOISY UNLEARNING WITH REGULARIZATION", flush=True)
            print("="*70, flush=True)
            print(f"Privacy target: (ε={epsilon:.4f}, δ={delta:.2e})", flush=True)
            print(f"Hyperparameters:", flush=True)
            print(f"  γ (learning rate) = {learning_rate:.6f}", flush=True)
            print(f"  λ (regularization) = {lambda_reg:.6f}", flush=True)
            print(f"  γλ = {learning_rate * lambda_reg:.6f} ∈ (0.5, 1.0) ✓", flush=True)
            print(f"Clipping thresholds:", flush=True)
            print(f"  C₀ (initial model) = {clip_norm_0:.6f}", flush=True)
            print(f"  C₁ (gradients) = {clip_norm_1:.6f}", flush=True)
            print(f"Required noise variance: σ² = {sigma_squared:.6e}", flush=True)
            print(f"Required noise std: σ = {np.sqrt(sigma_squared):.6e}", flush=True)
            print(f"Unlearning iterations: T = {num_iterations}", flush=True)
            print("="*70 + "\n", flush=True)

        # Step 0: Initial model clipping (C₀)
        if verbose:
            print("Clipping initial model parameters by C₀..."  , flush=True)
        clip_model_parameters(self.model, clip_norm_0)


        # Setup optimizer (no weight_decay - we do L2 regularization manually)
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=0.0)

        self.model.train()

        # Phase 1: Noisy fine-tuning with clipping and regularization
        unlearn_iterator = tqdm(
            range(num_iterations),
            desc="PHASE 1: Noisy Unlearning",
            unit="iter",
            disable=not verbose
        )

        for iteration in unlearn_iterator:
            iteration_loss = 0.0
            iteration_batches = 0

            for data, target in retain_loader:
                data = data.view(data.size(0), -1).to(self.device)
                target = target.to(self.device)

                # Step 1: Forward pass and compute loss
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, target)

                # Step 2: Backward pass
                loss.backward()

                # Step 3: Add L2 regularization term to gradients BEFORE clipping
                # This implements: grad := grad + λ·param
                add_regularization_to_gradients(self.model, lambda_reg)

                # Step 4: Clip gradients by C₁
                clip_gradients(self.model, clip_norm_1)

                # Step 5: Optimizer step
                optimizer.step()

                # Step 6: Add Gaussian noise to parameters
                add_noise_to_parameters(self.model, sigma_squared, self.device)

                iteration_loss += loss.item()
                iteration_batches += 1

                if iteration % 100 == 0:
                    mlflow.log_metric("unlearn_loss", loss.item(), step=iteration)

            avg_loss = iteration_loss / max(iteration_batches, 1)
            unlearn_iterator.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

        if verbose:
            print("\n✓ PHASE 1 COMPLETE: Noisy unlearning finished\n"  , flush=True)

        return self.model

    def post_unlearn_finetune(
            self,
            retain_loader: DataLoader,
            num_epochs: int,
            learning_rate: float,
            loss_fn: Optional[nn.Module] = None,
            weight_decay: float = 0.0,
            verbose: bool = True
    ) -> nn.Module:
        """
        Phase 2: Standard fine-tuning WITHOUT noise or clipping.

        This phase recovers model accuracy after the noisy unlearning phase.
        Uses standard SGD training (no gradient clipping, no noise addition).

        Args:
            retain_loader: DataLoader for data to retain (D_r)
            num_epochs: Number of fine-tuning epochs
            learning_rate: Learning rate
            loss_fn: Loss function (default: CrossEntropyLoss)
            weight_decay: L2 regularization weight decay (standard SGD)
            verbose: Print progress

        Returns:
            Fine-tuned model with recovered accuracy
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        if verbose:
            print("="*70, flush=True)
            print("PHASE 2: STANDARD FINE-TUNING (No Noise/Clipping)", flush=True)
            print("="*70, flush=True)
            print(f"Fine-tuning epochs: {num_epochs}", flush=True)
            print(f"Learning rate: {learning_rate:.6f}", flush=True)
            print(f"Weight decay: {weight_decay:.6f}", flush=True)
            print("="*70 + "\n", flush=True)

        # Setup optimizer for post-unlearning phase (standard SGD)
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.model.train()

        # Phase 2: Standard fine-tuning without any privacy mechanisms
        epoch_iterator = tqdm(
            range(num_epochs),
            desc="PHASE 2: Fine-Tuning",
            unit="epoch",
            disable=not verbose
        )

        for epoch in epoch_iterator:
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            # batch_iterator = tqdm(
            #     retain_loader,
            #     desc=f"  Epoch {epoch+1}/{num_epochs}",
            #     unit="batch",
            #     leave=False,
            #     disable=not verbose
            # )

            for data, target in retain_loader:
                data = data.view(data.size(0), -1).to(self.device)
                target = target.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, target)

                # Backward pass
                loss.backward()

                # Standard optimizer step (NO clipping, NO noise addition)
                optimizer.step()

                # Tracking
                epoch_loss += loss.item()
                with torch.no_grad():
                    predictions = output.argmax(dim=1)
                    epoch_correct += (predictions == target).sum().item()
                    epoch_total += target.size(0)

                mlflow.log_metric("finetune_loss", loss.item(), step=epoch)

            avg_loss = epoch_loss / max(len(retain_loader), 1)
            avg_acc = epoch_correct / max(epoch_total, 1)
            epoch_iterator.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{avg_acc:.4f}"
            })

            mlflow.log_metric("finetune_epoch_loss", avg_loss, step=epoch)
            mlflow.log_metric("finetune_epoch_accuracy", avg_acc, step=epoch)

        if verbose:
            print("\n✓ PHASE 2 COMPLETE: Fine-tuning finished\n", flush=True)
            print("="*70, flush=True)
            print("UNLEARNING PROCESS COMPLETE", flush=True)
            print("="*70 + "\n", flush=True)

        return self.model