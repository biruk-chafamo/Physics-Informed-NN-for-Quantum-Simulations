"""
Physics-informed loss functions for the Schrödinger equation.

Implements the residual losses for:
- Time-dependent Schrödinger equation (PDE residual)
- Initial conditions
- Boundary conditions
- Wave function normalization
"""

from typing import Tuple

import torch

from .config import DomainConfig
from .potentials import BasePotential


def compute_derivatives(
    phi_pred_rc: torch.Tensor,
    x_t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute first and second derivatives of the wave function using autograd.

    Computes ∂ψ/∂t and ∂²ψ/∂x² for both real and imaginary components,
    then combines them into complex derivatives.

    Args:
        phi_pred_rc: Network output of shape (N, 2) with (Re(ψ), Im(ψ)).
        x_t: Input coordinates of shape (N, 2) with (x, t). Must have requires_grad=True.

    Returns:
        Tuple of (d_t, d_xx):
            - d_t: Complex time derivative ∂ψ/∂t of shape (N,).
            - d_xx: Complex second spatial derivative ∂²ψ/∂x² of shape (N,).
    """
    phi_pred_r, phi_pred_c = phi_pred_rc[:, 0], phi_pred_rc[:, 1]

    # Compute gradients for real part
    d_grid_r = torch.autograd.grad(
        phi_pred_r, x_t,
        grad_outputs=torch.ones_like(phi_pred_r),
        create_graph=True,
        retain_graph=True
    )[0]
    d_x_r, d_t_r = d_grid_r[:, 0], d_grid_r[:, 1]

    d_xx_r = torch.autograd.grad(
        d_x_r, x_t,
        grad_outputs=torch.ones_like(d_x_r),
        create_graph=True,
        retain_graph=True
    )[0][:, 0]

    # Compute gradients for imaginary part
    d_grid_c = torch.autograd.grad(
        phi_pred_c, x_t,
        grad_outputs=torch.ones_like(phi_pred_c),
        create_graph=True,
        retain_graph=True
    )[0]
    d_x_c, d_t_c = d_grid_c[:, 0], d_grid_c[:, 1]

    d_xx_c = torch.autograd.grad(
        d_x_c, x_t,
        grad_outputs=torch.ones_like(d_x_c),
        create_graph=True,
        retain_graph=True
    )[0][:, 0]

    # Combine into complex derivatives
    d_t = d_t_r + 1j * d_t_c
    d_xx = d_xx_r + 1j * d_xx_c

    return d_t, d_xx


def physics_loss(
    phi: torch.Tensor,
    d_t: torch.Tensor,
    d_xx: torch.Tensor,
    potential_values: torch.Tensor,
    domain: DomainConfig
) -> torch.Tensor:
    """Compute the physics loss (PDE residual) for the Schrödinger equation.

    The time-dependent Schrödinger equation (in natural units, ℏ=1, m=1/2):
        i ∂ψ/∂t = -∂²ψ/∂x² + V(x)ψ

    Rearranged as residual:
        |i ∂ψ/∂t + (1/2)∂²ψ/∂x² - V(x)ψ| = 0

    Args:
        phi: Complex wave function of shape (N,).
        d_t: Time derivative ∂ψ/∂t of shape (N,).
        d_xx: Second spatial derivative ∂²ψ/∂x² of shape (N,).
        potential_values: Potential V(x) evaluated at grid points, shape (N,).
        domain: Domain configuration for reshaping.

    Returns:
        Scalar tensor with mean physics loss (excluding boundaries).
    """
    residual = torch.abs(1j * d_t + d_xx / 2 - potential_values * phi)

    # Reshape to (nx, nt) grid and exclude boundaries
    residual = residual.reshape(domain.nx, domain.nt)
    residual = residual[1:-1, :]  # Exclude x boundaries only, keep t=0

    return torch.mean(residual.reshape(-1))


def initial_condition_loss(
    phi_pred: torch.Tensor,
    phi_0: torch.Tensor,
    domain: DomainConfig
) -> torch.Tensor:
    """Compute the initial condition loss.

    Measures deviation from the target initial wave function at t=0:
        L_IC = mean(|ψ(x, t=0) - ψ₀(x)|)

    Args:
        phi_pred: Predicted complex wave function of shape (N,).
        phi_0: Target initial condition of shape (nx,).
        domain: Domain configuration for reshaping.

    Returns:
        Scalar tensor with mean initial condition loss.
    """
    phi_pred_t0 = phi_pred.reshape(domain.nx, domain.nt)[:, 0]
    return torch.mean(torch.abs(phi_pred_t0 - phi_0))


def boundary_condition_loss(
    phi_pred: torch.Tensor,
    domain: DomainConfig
) -> torch.Tensor:
    """Compute the boundary condition loss (Dirichlet BCs).

    Enforces ψ(x_min, t) = 0 and ψ(x_max, t) = 0:
        L_BC = mean(|ψ(x_min, t)|) + mean(|ψ(x_max, t)|)

    Args:
        phi_pred: Predicted complex wave function of shape (N,).
        domain: Domain configuration for reshaping.

    Returns:
        Scalar tensor with mean boundary condition loss.
    """
    phi_grid = phi_pred.reshape(domain.nx, domain.nt)
    phi_x_min = phi_grid[0, :]   # ψ(x_min, t)
    phi_x_max = phi_grid[-1, :]  # ψ(x_max, t)

    return torch.mean(torch.abs(phi_x_min)) + torch.mean(torch.abs(phi_x_max))


def normalization_loss(
    phi_pred: torch.Tensor,
    domain: DomainConfig
) -> torch.Tensor:
    """Compute the normalization loss.

    Enforces ∫|ψ(x, t)|² dx = 1 for all t:
        L_norm = mean(|∫|ψ|² dx - 1|)

    Args:
        phi_pred: Predicted complex wave function of shape (N,).
        domain: Domain configuration for reshaping and dx.

    Returns:
        Scalar tensor with mean normalization loss.
    """
    phi_grid = phi_pred.reshape(domain.nx, domain.nt)
    # Compute ∫|ψ|² dx for each time step
    norms = torch.sum(torch.abs(phi_grid) ** 2 * domain.dx, dim=0)

    return torch.mean(torch.abs(norms - 1.0))


def total_loss(
    phi_pred: torch.Tensor,
    d_t: torch.Tensor,
    d_xx: torch.Tensor,
    potential_values: torch.Tensor,
    phi_0: torch.Tensor,
    domain: DomainConfig,
    physics_weight: float = 15.0,
    initial_weight: float = 10.0,
    boundary_weight: float = 9.0,
    normalization_weight: float = 50.0,
    return_components: bool = False
) -> torch.Tensor:
    """Compute the total weighted loss.

    Total loss = α·L_physics + β·L_IC + γ·L_BC + δ·L_norm

    Args:
        phi_pred: Predicted complex wave function of shape (N,).
        d_t: Time derivative ∂ψ/∂t of shape (N,).
        d_xx: Second spatial derivative ∂²ψ/∂x² of shape (N,).
        potential_values: Potential V(x) evaluated at grid points, shape (N,).
        phi_0: Target initial condition of shape (nx,).
        domain: Domain configuration.
        physics_weight: Weight α for physics loss.
        initial_weight: Weight β for initial condition loss.
        boundary_weight: Weight γ for boundary condition loss.
        normalization_weight: Weight δ for normalization loss.
        return_components: If True, also return individual loss components.

    Returns:
        If return_components is False: Scalar tensor with total loss.
        If return_components is True: Tuple of (total_loss, loss_dict).
    """
    l_physics = physics_loss(phi_pred, d_t, d_xx, potential_values, domain)
    l_initial = initial_condition_loss(phi_pred, phi_0, domain)
    l_boundary = boundary_condition_loss(phi_pred, domain)
    l_norm = normalization_loss(phi_pred, domain)

    total = (
        physics_weight * l_physics +
        initial_weight * l_initial +
        boundary_weight * l_boundary +
        normalization_weight * l_norm
    )

    if return_components:
        loss_dict = {
            'physics': physics_weight * l_physics.item(),
            'initial': initial_weight * l_initial.item(),
            'boundary': boundary_weight * l_boundary.item(),
            'normalization': normalization_weight * l_norm.item(),
            'total': total.item()
        }
        return total, loss_dict

    return total


def create_initial_condition(
    x: torch.Tensor,
    amplitude: float = 10.0,
    center: float = 2.5,
    sigma: float = 0.2,
    dx: float = 0.005
) -> torch.Tensor:
    """Create a normalized Gaussian wave packet initial condition.

    ψ₀(x) = A·exp(-(x - center)² / (2σ²)), normalized so ∫|ψ₀|² dx = 1

    Args:
        x: Spatial coordinates of shape (nx,).
        amplitude: Amplitude before normalization.
        center: Center of the Gaussian wave packet.
        sigma: Width of the Gaussian.
        dx: Spatial step size for normalization integral.

    Returns:
        Normalized initial wave function of shape (nx,).
    """
    with torch.no_grad():
        phi_0_r = amplitude * torch.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        phi_0_c = 1j * amplitude * torch.exp(-((x - center + 0.1) ** 2) / (2 * sigma ** 2))
        phi_0 = phi_0_r + phi_0_c
        norm = torch.sum(torch.abs(phi_0) ** 2 * dx)
        phi_0 = phi_0 / torch.sqrt(norm)
    return phi_0


def create_grid(domain: DomainConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the spatial-temporal grid for training.

    Args:
        domain: Domain configuration.

    Returns:
        Tuple of (x, t, x_t):
            - x: Spatial coordinates of shape (nx,).
            - t: Temporal coordinates of shape (nt,).
            - x_t: Flattened grid of shape (nx*nt, 2) with requires_grad=True.
    """
    x = torch.linspace(domain.x_min, domain.x_max, domain.nx)
    t = torch.linspace(domain.t_min, domain.t_max, domain.nt)

    x_t = torch.vstack(torch.meshgrid(x, t, indexing='ij')).reshape(2, -1).T
    x_t.requires_grad = True

    return x, t, x_t
