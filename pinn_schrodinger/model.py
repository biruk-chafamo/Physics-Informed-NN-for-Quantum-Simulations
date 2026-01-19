"""
Neural network model for the PINN Schrödinger solver.

Implements a configurable feedforward network that outputs the real and
imaginary components of the wave function ψ(x, t).
"""

from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig, Config


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.

    Args:
        name: Activation function name ('tanh', 'silu').

    Returns:
        PyTorch activation module.

    Raises:
        ValueError: If activation name is not recognized.
    """
    activations = {
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Supported: {list(activations.keys())}")
    return activations[name.lower()]


class SchrodingerPINN(nn.Module):
    """Physics-Informed Neural Network for solving the 1D Schrödinger equation.

    The network takes (x, t) coordinates as input and outputs the real and
    imaginary parts of the wave function ψ(x, t).

    Architecture:
        Input: (x, t) -> 2 features
        Hidden layers: Configurable fully-connected layers with activation
        Output: (Re(ψ), Im(ψ))

    Attributes:
        config: Model configuration.
        model: Sequential neural network.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the PINN model.

        Args:
            config: Model configuration. If None, uses default configuration.
        """
        super().__init__()
        self.config = config or ModelConfig()

        layers = []
        input_dim = 2  # (x, t)

        # Build hidden layers
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation(self.config.activation))
            input_dim = hidden_dim

        # Output layer: real and imaginary parts of ψ
        layers.append(nn.Linear(input_dim, 2))

        self.model = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x_t: Input tensor of shape (N, 2) containing (x, t) coordinates.

        Returns:
            Tensor of shape (N, 2) containing (Re(ψ), Im(ψ)).
        """
        return self.model(x_t)

    def predict_complex(self, x_t: torch.Tensor) -> torch.Tensor:
        """Predict the complex wave function.

        Args:
            x_t: Input tensor of shape (N, 2) containing (x, t) coordinates.

        Returns:
            Complex tensor of shape (N,) containing ψ(x, t).
        """
        phi_rc = self.forward(x_t)
        return phi_rc[:, 0] + 1j * phi_rc[:, 1]

    def save(self, path: str, full_config: Optional[Config] = None) -> None:
        """Save model state to file.

        Args:
            path: File path to save the model checkpoint.
            full_config: Optional full Config object to save alongside model.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }
        if full_config is not None:
            checkpoint['full_config'] = full_config.to_dict()
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'SchrodingerPINN':
        """Load model from checkpoint file.

        Args:
            path: File path to the model checkpoint.
            device: Device to load the model onto. If None, uses CPU.

        Returns:
            Loaded SchrodingerPINN model.
        """
        device = device or torch.device('cpu')
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model = cls(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model

    @classmethod
    def load_with_config(cls, path: str, device: Optional[torch.device] = None) -> tuple['SchrodingerPINN', Optional[Config]]:
        """Load model and full config from checkpoint file.

        Args:
            path: File path to the model checkpoint.
            device: Device to load the model onto. If None, uses CPU.

        Returns:
            Tuple of (loaded model, full Config or None if not saved).
        """
        device = device or torch.device('cpu')
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model = cls(config=checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        full_config = None
        if 'full_config' in checkpoint:
            full_config = Config.from_dict(checkpoint['full_config'])

        return model, full_config

    def count_parameters(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"SchrodingerPINN(\n"
                f"  hidden_dims={self.config.hidden_dims},\n"
                f"  activation={self.config.activation},\n"
                f"  parameters={self.count_parameters()}\n"
                f")")
