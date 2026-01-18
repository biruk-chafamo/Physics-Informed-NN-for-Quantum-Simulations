"""
Potential functions for the Schrödinger equation.

Provides various potential types: Gaussian barrier, harmonic oscillator,
infinite square well, and custom user-defined potentials.
"""

from abc import ABC, abstractmethod
from typing import Callable

import torch


class BasePotential(ABC):
    """Abstract base class for potential functions.

    All potential classes must implement the __call__ method that computes
    V(x) for a given spatial coordinate tensor.
    """

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the potential at spatial coordinates.

        Args:
            x: Spatial coordinates tensor of shape (N,) or (N, 1).

        Returns:
            Potential values V(x) of the same shape as input.
        """
        pass


class GaussianPotential(BasePotential):
    """Gaussian barrier potential.

    V(x) = amplitude * exp(-(x - center)² / (2σ²))

    Attributes:
        amplitude: Height of the Gaussian barrier.
        center: Center position of the barrier.
        sigma: Width parameter of the Gaussian.
    """

    def __init__(self, amplitude: float = 20.0, center: float = 2.5, sigma: float = 0.5):
        """Initialize Gaussian potential.

        Args:
            amplitude: Height of the Gaussian barrier.
            center: Center position of the barrier.
            sigma: Width parameter of the Gaussian.
        """
        self.amplitude = amplitude
        self.center = center
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Gaussian potential at spatial coordinates."""
        return self.amplitude * torch.exp(-((x - self.center) ** 2) / (2 * self.sigma ** 2))

    def __repr__(self) -> str:
        return f"GaussianPotential(amplitude={self.amplitude}, center={self.center}, sigma={self.sigma})"


class HarmonicPotential(BasePotential):
    """Harmonic oscillator potential.

    V(x) = 0.5 * k * (x - x0)²

    Attributes:
        spring_constant: Spring constant k.
        center: Equilibrium position x0.
    """

    def __init__(self, spring_constant: float = 1.0, center: float = 2.5):
        """Initialize harmonic potential.

        Args:
            spring_constant: Spring constant k.
            center: Equilibrium position x0.
        """
        self.spring_constant = spring_constant
        self.center = center

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate harmonic potential at spatial coordinates."""
        return 0.5 * self.spring_constant * (x - self.center) ** 2

    def __repr__(self) -> str:
        return f"HarmonicPotential(k={self.spring_constant}, center={self.center})"


class InfiniteWellPotential(BasePotential):
    """Infinite square well potential.

    V(x) = 0 for left < x < right
    V(x) = large_value (simulating infinity) otherwise

    Note: True infinite potential is approximated with a large finite value
    to maintain numerical stability.

    Attributes:
        left: Left boundary of the well.
        right: Right boundary of the well.
        wall_height: Height of the potential walls (approximating infinity).
    """

    def __init__(self, left: float = 0.0, right: float = 5.0, wall_height: float = 1e6):
        """Initialize infinite well potential.

        Args:
            left: Left boundary of the well.
            right: Right boundary of the well.
            wall_height: Height of the potential walls.
        """
        self.left = left
        self.right = right
        self.wall_height = wall_height

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate infinite well potential at spatial coordinates."""
        inside_well = (x > self.left) & (x < self.right)
        potential = torch.where(inside_well, torch.zeros_like(x), torch.full_like(x, self.wall_height))
        return potential

    def __repr__(self) -> str:
        return f"InfiniteWellPotential(left={self.left}, right={self.right})"


class CustomPotential(BasePotential):
    """User-defined custom potential.

    Allows arbitrary potential functions V(x) defined as Python callables.

    Attributes:
        fn: Callable that takes a torch.Tensor and returns a torch.Tensor.
    """

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        """Initialize custom potential.

        Args:
            fn: A callable that computes V(x) given spatial coordinates x.
                Must accept a torch.Tensor and return a torch.Tensor.
        """
        self.fn = fn

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate custom potential at spatial coordinates."""
        return self.fn(x)

    def __repr__(self) -> str:
        return f"CustomPotential(fn={self.fn.__name__ if hasattr(self.fn, '__name__') else 'lambda'})"


def create_potential_from_config(config) -> BasePotential:
    """Factory function to create a potential from configuration.

    Args:
        config: PotentialConfig object with potential parameters.

    Returns:
        Appropriate BasePotential subclass instance.

    Raises:
        ValueError: If potential type is not recognized.
    """
    potential_type = config.type.lower()

    if potential_type == "gaussian":
        return GaussianPotential(
            amplitude=config.amplitude,
            center=config.center,
            sigma=config.sigma
        )
    elif potential_type == "harmonic":
        return HarmonicPotential(
            spring_constant=config.spring_constant,
            center=config.center
        )
    elif potential_type == "infinite_well":
        return InfiniteWellPotential(
            left=config.well_left,
            right=config.well_right
        )
    elif potential_type == "custom":
        if config.custom_fn is None:
            raise ValueError("Custom potential requires a custom_fn to be specified")
        return CustomPotential(fn=config.custom_fn)
    else:
        raise ValueError(f"Unknown potential type: {potential_type}. "
                        f"Supported types: gaussian, harmonic, infinite_well, custom")
