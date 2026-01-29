"""
Adaptive loss weight scheduling using Learning Rate Annealing (LRA).

Based on Wang, Teng & Perdikaris (2021) - gradient-based adaptive weight balancing.
Weights are adjusted so all loss terms contribute equally to parameter updates
by normalizing gradient magnitudes.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class LRAWeightScheduler:
    """Learning Rate Annealing weight scheduler for PINN loss balancing.

    Adjusts weights so that all loss terms contribute equally to parameter updates
    by normalizing their gradient magnitudes.

    Algorithm:
        For each loss L_i, compute gradient norm: g_i = ||∇_θ L_i||_2
        Adaptive weight: λ_i = max(g_j) / g_i
        With momentum: λ_i(t) = (1 - τ) * λ_i(t-1) + τ * (g_max / g_i)

    Attributes:
        loss_names: Names of the loss components to balance.
        tau: Momentum parameter for weight updates (0 < tau <= 1).
        update_freq: How often to update weights (in epochs).
        min_weight: Minimum allowed weight value.
        max_weight: Maximum allowed weight value.
        weights: Current weight dictionary.
        weight_history: History of weights for analysis.
    """

    def __init__(
        self,
        loss_names: List[str],
        tau: float = 0.9,
        update_freq: int = 100,
        min_weight: float = 0.1,
        max_weight: float = 100.0,
        initial_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize the LRA weight scheduler.

        Args:
            loss_names: Names of loss components (e.g., ['physics', 'initial', 'boundary', 'normalization']).
            tau: Momentum parameter (higher = faster adaptation, lower = smoother).
            update_freq: Update weights every N epochs.
            min_weight: Clamp weights to this minimum.
            max_weight: Clamp weights to this maximum.
            initial_weights: Optional initial weights. If None, all start at 1.0.
        """
        self.loss_names = loss_names
        self.tau = tau
        self.update_freq = update_freq
        self.min_weight = min_weight
        self.max_weight = max_weight

        if initial_weights is not None:
            self.weights = {name: initial_weights.get(name, 1.0) for name in loss_names}
        else:
            self.weights = {name: 1.0 for name in loss_names}

        self.weight_history: List[Dict[str, float]] = []

    def compute_gradient_norms(
        self,
        model: nn.Module,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute the L2 norm of gradients for each loss component.

        Args:
            model: The neural network model.
            losses: Dictionary of unweighted loss tensors (must have requires_grad=True).

        Returns:
            Dictionary mapping loss names to their gradient L2 norms.
        """
        grad_norms = {}

        for name in self.loss_names:
            if name not in losses:
                continue

            loss = losses[name]

            # Zero gradients
            model.zero_grad()

            # Compute gradients for this loss
            loss.backward(retain_graph=True)

            # Compute L2 norm of all parameter gradients
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            grad_norms[name] = total_norm

        # Clear gradients after computing norms
        model.zero_grad()

        return grad_norms

    def update(
        self,
        epoch: int,
        model: nn.Module,
        losses: Dict[str, torch.Tensor]
    ) -> bool:
        """Update weights if epoch matches update frequency.

        Args:
            epoch: Current epoch number.
            model: The neural network model.
            losses: Dictionary of unweighted loss tensors.

        Returns:
            True if weights were updated, False otherwise.
        """
        if epoch % self.update_freq != 0:
            return False

        # Compute gradient norms
        grad_norms = self.compute_gradient_norms(model, losses)

        if not grad_norms:
            return False

        # Find maximum gradient norm
        g_max = max(grad_norms.values())

        if g_max == 0:
            return False

        # Update weights with momentum
        for name in self.loss_names:
            if name not in grad_norms:
                continue

            g_i = grad_norms[name]
            if g_i == 0:
                continue

            # Target weight based on gradient balancing
            target_weight = g_max / g_i

            # Apply momentum update
            old_weight = self.weights[name]
            new_weight = (1 - self.tau) * old_weight + self.tau * target_weight

            # Clamp to valid range
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            self.weights[name] = new_weight

        # Record history
        self.weight_history.append({
            'epoch': epoch,
            **{f'weight_{name}': self.weights[name] for name in self.loss_names},
            **{f'grad_norm_{name}': grad_norms.get(name, 0.0) for name in self.loss_names}
        })

        return True

    def get_weights(self) -> Dict[str, float]:
        """Get current weight dictionary.

        Returns:
            Dictionary mapping loss names to their current weights.
        """
        return self.weights.copy()

    def get_weight_history(self) -> List[Dict[str, float]]:
        """Get the full weight update history.

        Returns:
            List of dictionaries containing weight values at each update.
        """
        return self.weight_history
