"""
PINN Schrödinger Solver

A Physics-Informed Neural Network for solving the 1D time-dependent Schrödinger equation.
"""

from .config import DomainConfig, ModelConfig, TrainingConfig, PotentialConfig, Config, load_config
from .model import SchrodingerPINN
from .potentials import GaussianPotential, HarmonicPotential, InfiniteWellPotential, CustomPotential
from .trainer import SchrodingerTrainer
from .visualization import (
    plot_initial_condition,
    plot_probability_density,
    plot_loss_curves,
    plot_boundary_check,
    plot_physics_residual,
    create_animation,
)

__version__ = "0.1.0"
__all__ = [
    "DomainConfig",
    "ModelConfig",
    "TrainingConfig",
    "PotentialConfig",
    "Config",
    "load_config",
    "SchrodingerPINN",
    "GaussianPotential",
    "HarmonicPotential",
    "InfiniteWellPotential",
    "CustomPotential",
    "SchrodingerTrainer",
    "plot_initial_condition",
    "plot_probability_density",
    "plot_loss_curves",
    "plot_boundary_check",
    "plot_physics_residual",
    "create_animation",
]
