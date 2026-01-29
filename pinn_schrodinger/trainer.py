"""
Training loop for the PINN Schrödinger solver.

Provides a clean trainer class with logging, checkpointing, and progress tracking.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch.optim import Adam

from .adaptive_weights import LRAWeightScheduler
from .config import Config, DomainConfig, TrainingConfig
from .model import SchrodingerPINN
from .physics import (
    compute_derivatives,
    total_loss,
    create_grid,
    create_initial_condition,
    physics_loss,
    initial_condition_loss,
    boundary_condition_loss,
    normalization_loss,
)
from .potentials import BasePotential


logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Container for training results.

    Attributes:
        losses: List of total loss values per epoch.
        loss_components: List of loss component dicts (physics, initial, boundary, norm).
        predictions: List of predicted wave functions at logged epochs.
        epochs_logged: List of epoch numbers where predictions were saved.
        training_time: Total training time in seconds.
        start_time: Training start timestamp (ISO format).
        end_time: Training end timestamp (ISO format).
        weight_history: History of adaptive weight updates (if enabled).
    """
    losses: List[float] = field(default_factory=list)
    loss_components: List[Dict[str, float]] = field(default_factory=list)
    predictions: List[torch.Tensor] = field(default_factory=list)
    epochs_logged: List[int] = field(default_factory=list)
    training_time: float = 0.0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    weight_history: List[Dict[str, float]] = field(default_factory=list)


class SchrodingerTrainer:
    """Trainer class for the PINN Schrödinger solver.

    Handles the training loop, logging, and checkpointing for solving the
    1D time-dependent Schrödinger equation using a physics-informed neural network.

    Attributes:
        model: The PINN model.
        potential: The potential function V(x).
        config: Training configuration.
        domain: Domain configuration.
        optimizer: PyTorch optimizer.
    """

    def __init__(
        self,
        model: SchrodingerPINN,
        potential: BasePotential,
        config: Config
    ):
        """Initialize the trainer.

        Args:
            model: The PINN model to train.
            potential: The potential function V(x).
            config: Full configuration object.
        """
        self.model = model
        self.potential = potential
        self.config = config
        self.domain = config.domain
        self.training_config = config.training

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.training_config.lr,
            amsgrad=True
        )

        # Initialize adaptive weight scheduler if enabled
        self.weight_scheduler: Optional[LRAWeightScheduler] = None
        if (self.training_config.adaptive_weights is not None and
            self.training_config.adaptive_weights.enabled):
            aw_config = self.training_config.adaptive_weights
            initial_weights = {
                'physics': self.training_config.physics_weight,
                'initial': self.training_config.initial_weight,
                'boundary': self.training_config.boundary_weight,
                'normalization': self.training_config.normalization_weight,
            }
            self.weight_scheduler = LRAWeightScheduler(
                loss_names=['physics', 'initial', 'boundary', 'normalization'],
                tau=aw_config.tau,
                update_freq=aw_config.update_freq,
                min_weight=aw_config.min_weight,
                max_weight=aw_config.max_weight,
                initial_weights=initial_weights,
            )

    def train(
        self,
        x_t: Optional[torch.Tensor] = None,
        phi_0: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> TrainingResult:
        """Run the training loop.

        Args:
            x_t: Grid coordinates of shape (N, 2). If None, created from config.
            phi_0: Initial condition of shape (nx,). If None, created from config.
            verbose: Whether to print progress to stdout.

        Returns:
            TrainingResult containing loss history and predictions.
        """
        # Create grid if not provided
        if x_t is None:
            x, t, x_t = create_grid(self.domain)
        else:
            x = torch.linspace(self.domain.x_min, self.domain.x_max, self.domain.nx)
            t = torch.linspace(self.domain.t_min, self.domain.t_max, self.domain.nt)

        # Create initial condition if not provided
        if phi_0 is None:
            phi_0 = create_initial_condition(
                x,
                amplitude=self.config.initial_condition.amplitude,
                center=self.config.initial_condition.center,
                sigma=self.config.initial_condition.sigma,
                dx=self.domain.dx
            )

        # Compute potential values on the grid
        with torch.no_grad():
            v = self.potential(x_t[:, 0])

        result = TrainingResult()

        # Start timing
        from datetime import datetime
        start_time = time.time()
        result.start_time = datetime.now().isoformat()

        # Get initial weights (may be updated by scheduler)
        current_weights = {
            'physics': self.training_config.physics_weight,
            'initial': self.training_config.initial_weight,
            'boundary': self.training_config.boundary_weight,
            'normalization': self.training_config.normalization_weight,
        }

        for epoch in range(self.training_config.epochs):
            self.optimizer.zero_grad()

            # Forward pass
            phi_pred_rc = self.model(x_t)
            phi_pred_r, phi_pred_c = phi_pred_rc[:, 0], phi_pred_rc[:, 1]
            phi_pred = phi_pred_r + 1j * phi_pred_c

            # Compute derivatives
            d_t, d_xx = compute_derivatives(phi_pred_rc, x_t)

            # Compute individual unweighted losses for adaptive weighting
            if self.weight_scheduler is not None:
                l_physics = physics_loss(phi_pred, d_t, d_xx, v, self.domain)
                l_initial = initial_condition_loss(phi_pred, phi_0, self.domain)
                l_boundary = boundary_condition_loss(phi_pred, self.domain)
                l_norm = normalization_loss(phi_pred, self.domain)

                unweighted_losses = {
                    'physics': l_physics,
                    'initial': l_initial,
                    'boundary': l_boundary,
                    'normalization': l_norm,
                }

                # Update weights if needed
                weights_updated = self.weight_scheduler.update(
                    epoch, self.model, unweighted_losses
                )
                if weights_updated:
                    current_weights = self.weight_scheduler.get_weights()
                    if verbose:
                        print(f"[LRA] Weights updated at epoch {epoch}:")
                        for name, weight in current_weights.items():
                            print(f"  {name}: {weight:.4f}")
                        print()

                # Compute weighted total loss
                loss = (
                    current_weights['physics'] * l_physics +
                    current_weights['initial'] * l_initial +
                    current_weights['boundary'] * l_boundary +
                    current_weights['normalization'] * l_norm
                )

                loss_dict = {
                    'physics': current_weights['physics'] * l_physics.item(),
                    'initial': current_weights['initial'] * l_initial.item(),
                    'boundary': current_weights['boundary'] * l_boundary.item(),
                    'normalization': current_weights['normalization'] * l_norm.item(),
                    'total': loss.item(),
                }
            else:
                # Standard fixed weights
                should_log = epoch % self.training_config.log_every == 0
                loss, loss_dict = total_loss(
                    phi_pred, d_t, d_xx, v, phi_0, self.domain,
                    physics_weight=current_weights['physics'],
                    initial_weight=current_weights['initial'],
                    boundary_weight=current_weights['boundary'],
                    normalization_weight=current_weights['normalization'],
                    return_components=True
                )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Record loss
            result.losses.append(loss.item())

            # Log progress
            should_log = epoch % self.training_config.log_every == 0
            if should_log:
                result.loss_components.append(loss_dict)
                result.predictions.append(phi_pred.detach().clone())
                result.epochs_logged.append(epoch)

                if verbose:
                    self._log_progress(epoch, loss_dict)

                logger.info(
                    f"Epoch {epoch}: total={loss_dict['total']:.3f}, "
                    f"physics={loss_dict['physics']:.3f}, "
                    f"initial={loss_dict['initial']:.3f}, "
                    f"boundary={loss_dict['boundary']:.3f}, "
                    f"norm={loss_dict['normalization']:.3f}"
                )

            # Checkpoint
            if (self.training_config.checkpoint_every > 0 and
                epoch > 0 and
                epoch % self.training_config.checkpoint_every == 0):
                checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
                self.model.save(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        # End timing
        from datetime import datetime
        end_time = time.time()
        result.training_time = end_time - start_time
        result.end_time = datetime.now().isoformat()

        # Record weight history if using adaptive weights
        if self.weight_scheduler is not None:
            result.weight_history = self.weight_scheduler.get_weight_history()

        return result

    def _log_progress(self, epoch: int, loss_dict: Dict[str, float]) -> None:
        """Print training progress to stdout.

        Args:
            epoch: Current epoch number.
            loss_dict: Dictionary of loss components.
        """
        print(f"Epoch {epoch}")
        print(f"  Physics loss:       {loss_dict['physics']:.3f}")
        print(f"  Initial cond loss:  {loss_dict['initial']:.3f}")
        print(f"  Boundary cond loss: {loss_dict['boundary']:.3f}")
        print(f"  Normalization loss: {loss_dict['normalization']:.3f}")
        print(f"  Total loss:         {loss_dict['total']:.3f}")
        print()
