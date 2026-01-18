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

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'nx': self.nx,
            'nt': self.nt,
            'x_min': self.x_min,
            'x_max': self.x_max,
            't_min': self.t_min,
            't_max': self.t_max,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DomainConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelConfig:
    """Configuration for the neural network architecture.

    Attributes:
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name ('leaky_relu', 'relu', 'tanh', 'silu').
    """
    hidden_dims: list[int] = field(default_factory=lambda: [32, 32, 32])
    activation: str = "leaky_relu"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**data)


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

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'epochs': self.epochs,
            'lr': self.lr,
            'log_every': self.log_every,
            'checkpoint_every': self.checkpoint_every,
            'physics_weight': self.physics_weight,
            'initial_weight': self.initial_weight,
            'boundary_weight': self.boundary_weight,
            'normalization_weight': self.normalization_weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**data)


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

    def to_dict(self) -> dict:
        """Convert to dictionary (excludes custom_fn which can't be serialized)."""
        return {
            'type': self.type,
            'amplitude': self.amplitude,
            'center': self.center,
            'sigma': self.sigma,
            'spring_constant': self.spring_constant,
            'well_left': self.well_left,
            'well_right': self.well_right,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PotentialConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k != 'custom_fn'})


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

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'type': self.type,
            'amplitude': self.amplitude,
            'center': self.center,
            'sigma': self.sigma,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InitialConditionConfig':
        """Create from dictionary."""
        return cls(**data)


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

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'domain': self.domain.to_dict(),
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'potential': self.potential.to_dict(),
            'initial_condition': self.initial_condition.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """Create from dictionary."""
        return cls(
            domain=DomainConfig.from_dict(data.get('domain', {})),
            model=ModelConfig.from_dict(data.get('model', {})),
            training=TrainingConfig.from_dict(data.get('training', {})),
            potential=PotentialConfig.from_dict(data.get('potential', {})),
            initial_condition=InitialConditionConfig.from_dict(data.get('initial_condition', {})),
        )

    def save_yaml(self, path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML file.
        """
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


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
