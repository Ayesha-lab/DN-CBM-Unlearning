"""
Training loop for binary classification probe.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os.path as osp


class BinaryProbeTrainer:
    """
    Trainer for binary classification probe.
    """
    
    def __init__(self, model, device='cuda', learning_rate=1e-3, l1_coeff=0.0):
        """
        Args:
            model: Binary linear probe model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            l1_coeff: L1 sparsity coefficient
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.l1_coeff = l1_coeff
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
    
    def _compute_l1_loss(self):
        """Compute L1 sparsity loss on model weights."""
        l1_loss = 0.0
        for param in self.model.parameters():
            l1_loss += torch.abs(param).sum()
        return l1_loss
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc="Training"):
            total_batches += 1
            
            # Move to device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.float().unsqueeze(1).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch_X)
            
            # Compute loss
            ce_loss = self.criterion(logits, batch_y)
            l1_loss = self._compute_l1_loss() if self.l1_coeff > 0 else 0.0
            loss = ce_loss + self.l1_coeff * l1_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / total_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """
        Evaluate on validation set.
        
        Returns:
            (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_X, batch_y in tqdm(val_loader, desc="Validating"):
            # Move to device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.float().unsqueeze(1).to(self.device)
            
            # Forward pass
            logits = self.model(batch_X)
            
            # Compute loss
            ce_loss = self.criterion(logits, batch_y)
            l1_loss = self._compute_l1_loss() if self.l1_coeff > 0 else 0.0
            loss = ce_loss + self.l1_coeff * l1_loss
            total_loss += loss.item()
            
            # Compute accuracy
            preds = (logits > 0).long().squeeze()
            total_correct += (preds == batch_y.squeeze().long()).sum().item()
            total_samples += batch_y.shape[0]
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        self.val_losses.append(avg_loss)
        self.val_accs.append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50, val_freq=5):
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
            train_loss = self.train_epoch(train_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f}", end="")
            
            if (epoch + 1) % val_freq == 0:
                val_loss, val_acc = self.evaluate(val_loader)
                print(f" | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f}")
                
                # Track best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
            else:
                print()
        
        print(f"\n✓ Training complete!")
        print(f"✓ Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        return best_val_acc
    
    def get_model_weights(self):
        """Get model weights for interpretation."""
        return self.model.linear.weight.detach().cpu().squeeze()