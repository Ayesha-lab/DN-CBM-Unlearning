"""
Training loop for binary classification probe.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class BinaryProbeTrainer:
    """
    Trainer for binary classification probe.
    """

    def __init__(
        self,
        model,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        l1_coeff: float = 0.0,
        mlflow_logger=None,
        sparsity_eps: float = 1e-6,
    ):
        """
        Args:
            model: Binary linear probe model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            l1_coeff: L1 sparsity coefficient
            mlflow_logger: MLflowLogger from mlflow_utils (or None)
            sparsity_eps: Threshold for counting non-zero weights in sparsity metrics
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.l1_coeff = l1_coeff
        self.mlflow_logger = mlflow_logger
        self.sparsity_eps = sparsity_eps

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def _compute_l1_loss(self) -> torch.Tensor:
        """Compute L1 sparsity loss on model weights."""
        l1_loss = 0.0
        for param in self.model.parameters():
            l1_loss = l1_loss + torch.abs(param).sum()
        return l1_loss

    @torch.no_grad()
    def _weight_sparsity_metrics(self):
        """
        Sparsity metrics for the probe weights.
        Uses the linear layer weight: shape [1, n_concepts] (or [n_concepts]).
        """
        w = self.model.linear.weight.detach()
        w_abs = w.abs()

        nnz = (w_abs > self.sparsity_eps).sum().item()
        total = w.numel()
        nnz_frac = float(nnz) / float(total) if total > 0 else 0.0

        l1 = w_abs.sum().item()
        l2 = (w ** 2).sum().sqrt().item()

        return {
            "probe/weight_nnz": float(nnz),
            "probe/weight_nnz_frac": float(nnz_frac),
            "probe/weight_l1": float(l1),
            "probe/weight_l2": float(l2),
        }

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Returns:
            dict of averaged metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_correct = 0
        total_samples = 0
        total_batches = 0

        for batch_X, batch_y in tqdm(train_loader, desc="Training"):
            total_batches += 1

            batch_X = batch_X.to(self.device)
            batch_y = batch_y.float().unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_X)

            ce_loss = self.criterion(logits, batch_y)
            l1_loss = self._compute_l1_loss() if self.l1_coeff > 0 else 0.0
            loss = ce_loss + self.l1_coeff * l1_loss

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_ce += float(ce_loss.item())

            preds = (logits > 0).long()
            total_correct += (preds == batch_y.long()).sum().item()
            total_samples += batch_y.shape[0]

        avg_loss = total_loss / max(1, total_batches)
        avg_ce = total_ce / max(1, total_batches)
        acc = total_correct / max(1, total_samples)

        self.train_losses.append(avg_loss)

        return {
            "train/loss_total": float(avg_loss),
            "train/loss_ce": float(avg_ce),
            "train/acc_top1": float(acc),
        }

    @torch.no_grad()
    def evaluate(self, loader, split_name: str = "val"):
        """
        Evaluate on a loader.

        Returns:
            dict of averaged metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_correct = 0
        total_samples = 0
        total_batches = 0

        for batch_X, batch_y in tqdm(loader, desc=f"Evaluating ({split_name})"):
            total_batches += 1

            batch_X = batch_X.to(self.device)
            batch_y = batch_y.float().unsqueeze(1).to(self.device)

            logits = self.model(batch_X)

            ce_loss = self.criterion(logits, batch_y)
            l1_loss = self._compute_l1_loss() if self.l1_coeff > 0 else 0.0
            loss = ce_loss + self.l1_coeff * l1_loss

            total_loss += float(loss.item())
            total_ce += float(ce_loss.item())

            preds = (logits > 0).long()
            total_correct += (preds == batch_y.long()).sum().item()
            total_samples += batch_y.shape[0]

        avg_loss = total_loss / max(1, total_batches)
        avg_ce = total_ce / max(1, total_batches)
        acc = total_correct / max(1, total_samples)

        if split_name == "val":
            self.val_losses.append(avg_loss)
            self.val_accs.append(acc)

        return {
            f"{split_name}/loss_total": float(avg_loss),
            f"{split_name}/loss_ce": float(avg_ce),
            f"{split_name}/acc_top1": float(acc),
        }

    def train(self, train_loader, val_loader, num_epochs: int = 50, val_freq: int = 5):
        """
        Train for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            val_freq: Validation frequency (every N epochs)
        """
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader)
            sparsity_metrics = self._weight_sparsity_metrics()

            # log train + sparsity each epoch
            if self.mlflow_logger is not None:
                self.mlflow_logger.log_metrics({**train_metrics, **sparsity_metrics}, step=epoch)

            msg = (
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['train/loss_total']:.6f} | "
                f"Train Acc: {train_metrics['train/acc_top1']:.4f}"
            )

            if (epoch + 1) % val_freq == 0:
                val_metrics = self.evaluate(val_loader, split_name="val")

                if self.mlflow_logger is not None:
                    self.mlflow_logger.log_metrics(val_metrics, step=epoch)

                msg += (
                    f" | Val Loss: {val_metrics['val/loss_total']:.6f} | "
                    f"Val Acc: {val_metrics['val/acc_top1']:.4f}"
                )

                if val_metrics["val/acc_top1"] > best_val_acc:
                    best_val_acc = val_metrics["val/acc_top1"]
                    best_epoch = epoch + 1

            print(msg)

        print(f"\n✓ Training complete!")
        print(f"✓ Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metric("val/best_acc_top1", float(best_val_acc))

        return best_val_acc

    def get_model_weights(self):
        """Get model weights for interpretation."""
        return self.model.linear.weight.detach().cpu().squeeze()