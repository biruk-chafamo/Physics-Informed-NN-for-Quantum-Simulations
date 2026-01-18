"""
Configuration dataclasses for the PINN Schrödinger solver.

Provides structured configuration for domain, model, training, and potential settings.
Supports loading from YAML files.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import yaml


@dataclass
class DomainConfig:
    """Configuration for the spatial-temporal domain.

    Attributes:
        nx: Number of spatial discretization points.
        nt: Number of temporal discretization points.
        x_min: Left boundary of spatial domain.
        x_max: Right boundary of spatial domain.
        t_min: Initial time.
        t_max: Final time.
    """
    nx: int = 100
    nt: int = 100
    x_min: float = 0.0
    x_max: float = 5.01
    t_min: float = 0.0
    t_max: float = 5.01

    @property
    def dx(self) -> float:
        """Spatial step size."""
        return (self.x_max - self.x_min) / self.nx

    @property
    def dt(self) -> float:
        """Temporal step size."""
        return (self.t_max - self.t_min) / self.nt


@dataclass
class ModelConfig:
    """Configuration for the neural network architecture.

    Attributes:
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name ('leaky_relu', 'relu', 'tanh', 'silu').
    """
    hidden_dims: list[int] = field(default_factory=lambda: [32, 32, 32])
    activation: str = "leaky_relu"


@dataclass
class TrainingConfig:
    """Configuration for the training process.

    Attributes:
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        log_every: Frequency of logging (in epochs).
        checkpoint_every: Frequency of model checkpointing (in epochs). 0 to disable.
        physics_weight: Weight for physics (PDE residual) loss term (α).
        initial_weight: Weight for initial condition loss term (β).
        boundary_weight: Weight for boundary condition loss term (γ).
        normalization_weight: Weight for normalization loss term (δ).
    """
    epochs: int = 10001
    lr: float = 1e-3
    log_every: int = 1000
    checkpoint_every: int = 0
    physics_weight: float = 15.0
    initial_weight: float = 10.0
    boundary_weight: float = 9.0
    normalization_weight: float = 50.0


@dataclass
class PotentialConfig:
    """Configuration for the potential function.

    Attributes:
        type: Potential type ('gaussian', 'harmonic', 'infinite_well', 'custom').
        amplitude: Amplitude/strength of the potential.
        center: Center position of the potential.
        sigma: Width parameter (for Gaussian potential).
        spring_constant: Spring constant k (for harmonic potential).
        well_left: Left boundary (for infinite well).
        well_right: Right boundary (for infinite well).
        custom_fn: Custom potential function V(x) (for custom potential).
    """
    type: str = "gaussian"
    amplitude: float = 20.0
    center: float = 2.5
    sigma: float = 0.5
    spring_constant: float = 1.0
    well_left: float = 0.0
    well_right: float = 5.0
    custom_fn: Optional[Callable] = None


@dataclass
class InitialConditionConfig:
    """Configuration for the initial wave function.

    Attributes:
        type: Initial condition type ('gaussian').
        amplitude: Amplitude of the Gaussian wave packet.
        center: Center position of the wave packet.
        sigma: Width of the wave packet.
    """
    type: str = "gaussian"
    amplitude: float = 10.0
    center: float = 2.5
    sigma: float = 0.2


@dataclass
class Config:
    """Complete configuration for the PINN solver.

    Attributes:
        domain: Domain configuration.
        model: Model architecture configuration.
        training: Training configuration.
        potential: Potential function configuration.
        initial_condition: Initial condition configuration.
    """
    domain: DomainConfig = field(default_factory=DomainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    potential: PotentialConfig = field(default_factory=PotentialConfig)
    initial_condition: InitialConditionConfig = field(default_factory=InitialConditionConfig)


def load_config(yaml_path: str) -> Config:
    """Load configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Config object with loaded settings.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    config = Config()

    if 'domain' in data:
        config.domain = DomainConfig(**data['domain'])

    if 'model' in data:
        config.model = ModelConfig(**data['model'])

    if 'training' in data:
        config.training = TrainingConfig(**data['training'])

    if 'potential' in data:
        config.potential = PotentialConfig(**{
            k: v for k, v in data['potential'].items()
            if k != 'custom_fn'  # custom_fn can't be loaded from YAML
        })

    if 'initial_condition' in data:
        config.initial_condition = InitialConditionConfig(**data['initial_condition'])

    return config
