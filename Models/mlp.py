import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron for MNIST classification.

    Args:
        input_dim: Input feature dimension (784 for flattened MNIST)
        output_dim: Number of output classes (10 for MNIST)
        hidden_dim: Hidden layer dimension (default: 32)
    """
    def __init__(self, input_dim, output_dim, hidden_dim = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)