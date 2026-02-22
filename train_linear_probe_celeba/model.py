"""
Model architecture for binary classification probe.
"""

import torch
import torch.nn as nn


class BinaryLinearProbe(nn.Module):
    """
    Linear probe for binary classification on concept activations.
    """
    
    def __init__(self, n_concepts, use_bias=False):
        """
        Args:
            n_concepts: Number of input concept dimensions
            use_bias: Whether to use bias term
        """
        super().__init__()
        self.linear = nn.Linear(n_concepts, 1, bias=use_bias)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, n_concepts]
        
        Returns:
            logits: [batch_size, 1]
        """
        return self.linear(x)


def create_model(n_concepts, device='cuda'):
    """
    Create and initialize binary probe model.
    """
    model = BinaryLinearProbe(n_concepts=n_concepts)
    model = model.to(device)
    return model