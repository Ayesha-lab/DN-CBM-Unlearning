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
        w = self.model.linear.weight.detach()
        w_abs = w.abs()
        nnz = (w_abs > self.sparsity_eps).sum().item()
        total = w.numel()
        return {
            "probe/weight_nnz":      float(nnz),
            "probe/weight_nnz_frac": float(nnz / total) if total > 0 else 0.0,
            "probe/weight_l1":       float(w_abs.sum().item()),
            "probe/weight_l2":       float((w ** 2).sum().sqrt().item()),
        }

    def _binary_metrics(self, all_preds, all_labels):
        """Compute acc, balanced acc, TPR, TNR from accumulated tensors."""
        correct = (all_preds == all_labels).sum().item()
        total   = all_labels.shape[0]
        acc     = correct / max(1, total)

        pos_mask = (all_labels == 1)
        neg_mask = (all_labels == 0)
        tpr = (all_preds[pos_mask] == 1).sum().item() / max(1, pos_mask.sum().item())
        tnr = (all_preds[neg_mask] == 0).sum().item() / max(1, neg_mask.sum().item())
        bal_acc = (tpr + tnr) / 2.0

        return float(acc), float(bal_acc), float(tpr), float(tnr)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_ce   = 0.0
        total_batches = 0
        all_preds  = []
        all_labels = []

        for batch_X, batch_y in tqdm(train_loader, desc="Training"):
            total_batches += 1
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.float().unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()
            logits  = self.model(batch_X)
            ce_loss = self.criterion(logits, batch_y)
            l1_loss = self._compute_l1_loss() if self.l1_coeff > 0 else 0.0
            loss    = ce_loss + self.l1_coeff * l1_loss
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_ce   += float(ce_loss.item())
            all_preds.append((logits > 0).long().squeeze(1).cpu())
            all_labels.append(batch_y.long().squeeze(1).cpu())

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc, bal_acc, tpr, tnr = self._binary_metrics(all_preds, all_labels)

        avg_loss = total_loss / max(1, total_batches)
        self.train_losses.append(avg_loss)

        return {
            "train/loss_total": float(avg_loss),
            "train/loss_ce":    float(total_ce / max(1, total_batches)),
            "train/acc":        acc,
            "train/bal_acc":    bal_acc,
            "train/tpr":        tpr,
            "train/tnr":        tnr,
        }

    @torch.no_grad()
    def evaluate(self, loader, split_name: str = "val"):
        self.model.eval()
        total_loss = 0.0
        total_ce   = 0.0
        total_batches = 0
        all_preds  = []
        all_labels = []

        for batch_X, batch_y in tqdm(loader, desc=f"Evaluating ({split_name})"):
            total_batches += 1
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.float().unsqueeze(1).to(self.device)

            logits  = self.model(batch_X)
            ce_loss = self.criterion(logits, batch_y)
            l1_loss = self._compute_l1_loss() if self.l1_coeff > 0 else 0.0
            loss    = ce_loss + self.l1_coeff * l1_loss

            total_loss += float(loss.item())
            total_ce   += float(ce_loss.item())
            all_preds.append((logits > 0).long().squeeze(1).cpu())
            all_labels.append(batch_y.long().squeeze(1).cpu())

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc, bal_acc, tpr, tnr = self._binary_metrics(all_preds, all_labels)

        avg_loss = total_loss / max(1, total_batches)

        if split_name == "val":
            self.val_losses.append(avg_loss)
            self.val_accs.append(bal_acc)

        return {
            f"{split_name}/loss_total": float(avg_loss),
            f"{split_name}/loss_ce":    float(total_ce / max(1, total_batches)),
            f"{split_name}/acc":        acc,
            f"{split_name}/bal_acc":    bal_acc,
            f"{split_name}/tpr":        tpr,
            f"{split_name}/tnr":        tnr,
        }

    def train(self, train_loader, val_loader, num_epochs: int = 50, val_freq: int = 5):
        best_val_bal_acc = 0.0
        best_epoch = 0

        for epoch in range(num_epochs):
            train_metrics   = self.train_epoch(train_loader)
            sparsity_metrics = self._weight_sparsity_metrics()

            if self.mlflow_logger is not None:
                self.mlflow_logger.log_metrics({**train_metrics, **sparsity_metrics}, step=epoch)

            msg = (
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['train/loss_total']:.6f} | "
                f"Train Acc: {train_metrics['train/acc']:.4f} | "
                f"Train BalAcc: {train_metrics['train/bal_acc']:.4f}"
            )

            if (epoch + 1) % val_freq == 0:
                val_metrics = self.evaluate(val_loader, split_name="val")

                if self.mlflow_logger is not None:
                    self.mlflow_logger.log_metrics(val_metrics, step=epoch)

                msg += (
                    f" | Val Loss: {val_metrics['val/loss_total']:.6f} | "
                    f"Val Acc: {val_metrics['val/acc']:.4f} | "
                    f"Val BalAcc: {val_metrics['val/bal_acc']:.4f}"
                )

                if val_metrics["val/bal_acc"] > best_val_bal_acc:
                    best_val_bal_acc = val_metrics["val/bal_acc"]
                    best_epoch = epoch + 1

            print(msg)

        print(f"\n✓ Training complete!")
        print(f"✓ Best val balanced accuracy: {best_val_bal_acc:.4f} at epoch {best_epoch}")

        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metric("val/best_bal_acc", float(best_val_bal_acc))

        return best_val_bal_acc

    def get_model_weights(self):
        """Get model weights for interpretation."""
        return self.model.linear.weight.detach().cpu().squeeze()