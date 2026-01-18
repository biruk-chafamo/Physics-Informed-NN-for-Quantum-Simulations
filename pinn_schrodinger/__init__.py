"""
PINN Schrödinger Solver

A Physics-Informed Neural Network for solving the 1D time-dependent Schrödinger equation.
"""

from .config import DomainConfig, ModelConfig, TrainingConfig, PotentialConfig, Config, load_config
from .model import SchrodingerPINN
from .potentials import GaussianPotential, HarmonicPotential, InfiniteWellPotential, CustomPotential
from .trainer import SchrodingerTrainer, TrainingResult
from .run_manager import RunManager
from .inference import PINNPredictor, PredictionResult
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
    # Config
    "DomainConfig",
    "ModelConfig",
    "TrainingConfig",
    "PotentialConfig",
    "Config",
    "load_config",
    # Model
    "SchrodingerPINN",
    # Potentials
    "GaussianPotential",
    "HarmonicPotential",
    "InfiniteWellPotential",
    "CustomPotential",
    # Training
    "SchrodingerTrainer",
    "TrainingResult",
    # Run management
    "RunManager",
    # Inference
    "PINNPredictor",
    "PredictionResult",
    # Visualization
    "plot_initial_condition",
    "plot_probability_density",
    "plot_loss_curves",
    "plot_boundary_check",
    "plot_physics_residual",
    "create_animation",
]
