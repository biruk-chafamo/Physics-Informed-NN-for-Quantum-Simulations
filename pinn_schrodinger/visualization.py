"""
Visualization utilities for the PINN Schrödinger solver.

Provides plotting functions for:
- Initial conditions and potentials
- Probability density evolution
- Training loss curves
- Boundary condition verification
- Animated wave function evolution
"""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
import torch

from .config import DomainConfig
from .potentials import BasePotential


def plot_initial_condition(
    x: torch.Tensor,
    phi_0: torch.Tensor,
    potential: BasePotential,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot the initial wave function and potential.

    Creates a two-panel figure showing:
    - Top: Probability density |ψ₀(x)|² at t=0
    - Bottom: Potential V(x)

    Args:
        x: Spatial coordinates of shape (nx,).
        phi_0: Initial wave function of shape (nx,).
        potential: Potential function.
        save_path: If provided, saves figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    x_np = x.detach().numpy()
    phi_0_np = phi_0.detach().numpy()

    with torch.no_grad():
        v_np = potential(x).detach().numpy()

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot initial condition
    axs[0].plot(x_np, np.abs(phi_0_np) ** 2, 'b-', linewidth=1.5)
    axs[0].set_title('Initial Condition')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel(r'$|\psi_0(x)|^2$')
    axs[0].grid(True, alpha=0.3)

    # Plot potential
    axs[1].plot(x_np, v_np, 'r-', linewidth=1.5)
    axs[1].set_title('Potential V(x)')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('V(x)')
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_probability_density(
    x: torch.Tensor,
    t: torch.Tensor,
    phi_pred: torch.Tensor,
    phi_0: Optional[torch.Tensor] = None,
    domain: Optional[DomainConfig] = None,
    num_snapshots: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot probability density |ψ(x,t)|² at multiple time snapshots.

    Args:
        x: Spatial coordinates of shape (nx,).
        t: Temporal coordinates of shape (nt,).
        phi_pred: Predicted wave function of shape (nx*nt,) or (nx, nt).
        phi_0: Optional initial condition to overlay.
        domain: Domain configuration. If None, inferred from x and t.
        num_snapshots: Number of time snapshots to plot.
        save_path: If provided, saves figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    x_np = x.detach().numpy()
    t_np = t.detach().numpy()

    if domain is None:
        nx, nt = len(x), len(t)
    else:
        nx, nt = domain.nx, domain.nt

    # Handle complex tensors
    if torch.is_complex(phi_pred):
        phi_np = phi_pred.detach().numpy()
    else:
        phi_np = phi_pred.detach().numpy()

    phi_grid = phi_np.reshape(nx, nt)
    prob_density = np.abs(phi_grid) ** 2

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot initial condition if provided
    if phi_0 is not None:
        phi_0_np = phi_0.detach().numpy()
        ax.plot(x_np, np.abs(phi_0_np) ** 2, 'k--', linewidth=2, label=r'$|\psi_0|^2$')

    # Plot snapshots at different times
    cmap = plt.cm.viridis
    for i in range(num_snapshots):
        t_idx = int(nt * i / num_snapshots)
        color = cmap(i / num_snapshots)
        ax.plot(x_np, prob_density[:, t_idx], color=color,
                linewidth=1, label=f't={t_np[t_idx]:.2f}')

    ax.set_xlabel('x')
    ax.set_ylabel(r'$|\psi(x,t)|^2$')
    ax.set_title('Probability Density Evolution')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_loss_curves(
    losses: List[float],
    loss_components: Optional[List[dict]] = None,
    epochs_logged: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training loss curves.

    Args:
        losses: Total loss per epoch.
        loss_components: Optional list of loss component dicts.
        epochs_logged: Epochs where loss_components were recorded.
        save_path: If provided, saves figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot total loss
    axs[0].plot(losses, 'b-', linewidth=0.5, alpha=0.7)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Total Loss')
    axs[0].set_title('Training Loss')
    axs[0].set_yscale('log')
    axs[0].grid(True, alpha=0.3)

    # Plot loss components if available
    if loss_components and epochs_logged:
        physics = [lc['physics'] for lc in loss_components]
        initial = [lc['initial'] for lc in loss_components]
        boundary = [lc['boundary'] for lc in loss_components]
        norm = [lc['normalization'] for lc in loss_components]

        axs[1].plot(epochs_logged, physics, 'r-o', label='Physics', markersize=3)
        axs[1].plot(epochs_logged, initial, 'g-s', label='Initial Cond', markersize=3)
        axs[1].plot(epochs_logged, boundary, 'b-^', label='Boundary Cond', markersize=3)
        axs[1].plot(epochs_logged, norm, 'm-d', label='Normalization', markersize=3)
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss (weighted)')
        axs[1].set_title('Loss Components')
        axs[1].legend()
        axs[1].set_yscale('log')
        axs[1].grid(True, alpha=0.3)
    else:
        axs[1].text(0.5, 0.5, 'No component data', ha='center', va='center')
        axs[1].set_title('Loss Components')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_boundary_check(
    t: torch.Tensor,
    phi_pred: torch.Tensor,
    domain: DomainConfig,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot wave function values at boundaries to verify boundary conditions.

    Args:
        t: Temporal coordinates of shape (nt,).
        phi_pred: Predicted wave function of shape (nx*nt,) or (nx, nt).
        domain: Domain configuration.
        save_path: If provided, saves figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    t_np = t.detach().numpy()

    # Handle complex tensors
    if torch.is_complex(phi_pred):
        phi_np = phi_pred.detach().numpy()
    else:
        phi_np = phi_pred.detach().numpy()

    phi_grid = phi_np.reshape(domain.nx, domain.nt)
    prob_density = np.abs(phi_grid) ** 2

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t_np, prob_density[0, :], 'b-', linewidth=1.5, label=f'x = {domain.x_min:.2f} (left)')
    ax.plot(t_np, prob_density[-1, :], 'r--', linewidth=1.5, label=f'x = {domain.x_max:.2f} (right)')
    ax.plot(t_np, prob_density[domain.nx // 2, :], 'g-.', linewidth=1.5, label=f'x = {(domain.x_min + domain.x_max) / 2:.2f} (center)')

    ax.set_xlabel('t')
    ax.set_ylabel(r'$|\psi(x,t)|^2$')
    ax.set_title('Boundary Conditions Check')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_physics_residual(
    x: torch.Tensor,
    t: torch.Tensor,
    physics_loss_grid: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot the physics (PDE residual) loss as a heatmap.

    Args:
        x: Spatial coordinates of shape (nx,).
        t: Temporal coordinates of shape (nt,).
        physics_loss_grid: Physics residual of shape (nx, nt).
        save_path: If provided, saves figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    x_np = x.detach().numpy()
    t_np = t.detach().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.pcolormesh(t_np, x_np, physics_loss_grid, shading='auto', cmap='hot')
    plt.colorbar(im, ax=ax, label='PDE Residual')

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title('Physics Loss Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_animation(
    x: torch.Tensor,
    t: torch.Tensor,
    phi_pred: torch.Tensor,
    phi_0: Optional[torch.Tensor] = None,
    domain: Optional[DomainConfig] = None,
    num_frames: int = 100,
    save_path: Optional[Union[str, Path]] = None,
    fps: int = 20,
    dpi: int = 100,
    figsize: tuple = (10, 6),
    show_time: bool = True,
    title: str = 'Predicted Probability Density'
) -> FuncAnimation:
    """Create an animation of the probability density evolution over time.

    Generates an animated visualization showing |ψ(x,t)|² as it evolves
    from the initial condition. Can be saved as GIF or MP4.

    Args:
        x: Spatial coordinates of shape (nx,).
        t: Temporal coordinates of shape (nt,).
        phi_pred: Predicted wave function of shape (nx*nt,) or (nx, nt).
        phi_0: Optional initial condition to overlay as reference.
        domain: Domain configuration. If None, inferred from x and t.
        num_frames: Number of frames in the animation.
        save_path: Path to save the animation. Supports .gif and .mp4 extensions.
            If None, animation is not saved.
        fps: Frames per second for the saved animation.
        dpi: Resolution of the saved animation.
        figsize: Figure size as (width, height) in inches.
        show_time: Whether to display the current time value.
        title: Title for the plot.

    Returns:
        FuncAnimation object. Call plt.show() to display interactively.

    Example:
        >>> anim = create_animation(x, t, phi_pred, phi_0,
        ...                         save_path='wave_evolution.gif')
        >>> plt.show()  # For interactive display
    """
    # Convert tensors to numpy
    x_np = x.detach().numpy() if hasattr(x, 'detach') else np.asarray(x)
    t_np = t.detach().numpy() if hasattr(t, 'detach') else np.asarray(t)

    if domain is None:
        nx, nt = len(x), len(t)
    else:
        nx, nt = domain.nx, domain.nt

    # Handle complex tensors
    if hasattr(phi_pred, 'detach'):
        phi_np = phi_pred.detach().numpy()
    else:
        phi_np = np.asarray(phi_pred)

    phi_grid = phi_np.reshape(nx, nt)
    prob_density = np.abs(phi_grid) ** 2

    # Process initial condition if provided
    phi_0_density = None
    if phi_0 is not None:
        phi_0_np = phi_0.detach().numpy() if hasattr(phi_0, 'detach') else np.asarray(phi_0)
        phi_0_density = np.abs(phi_0_np) ** 2

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot initial condition as static reference
    if phi_0_density is not None:
        ax.plot(x_np, phi_0_density, label=r'$|\psi_0|^2$ (initial)',
                linestyle='--', color='blue', linewidth=2, zorder=10)

    # Initialize the animated line
    line, = ax.plot([], [], linestyle='-', linewidth=2, color='red',
                    label=r'$|\psi(x,t)|^2$')

    # Configure axes
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel(r'$|\psi(x)|^2$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', prop={'size': 10})
    ax.set_xlim(x_np.min(), x_np.max())
    ax.set_ylim(0, prob_density.max() * 1.1)
    ax.grid(True, alpha=0.3)

    # Add time display
    time_text = None
    if show_time:
        time_text = ax.text(
            0.02, 0.95, '', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    def init():
        """Initialize animation."""
        line.set_data([], [])
        if time_text is not None:
            time_text.set_text('')
            return line, time_text
        return (line,)

    def animate(frame):
        """Update animation for each frame."""
        # Map frame number to time index
        t_idx = int((nt - 1) * frame / (num_frames - 1)) if num_frames > 1 else 0
        t_idx = min(t_idx, nt - 1)  # Ensure we don't exceed bounds

        # Update line data
        line.set_data(x_np, prob_density[:, t_idx])

        # Update time display
        if time_text is not None:
            current_time = t_np[t_idx]
            time_text.set_text(f't = {current_time:.3f}')
            return line, time_text
        return (line,)

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=num_frames, interval=1000 // fps,
        blit=True, repeat=True
    )

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        extension = save_path.suffix.lower()

        if extension == '.gif':
            writer = PillowWriter(fps=fps)
            anim.save(str(save_path), writer=writer, dpi=dpi)
        elif extension in ('.mp4', '.mov'):
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(str(save_path), writer=writer, dpi=dpi)
        else:
            # Default to GIF for unknown extensions
            writer = PillowWriter(fps=fps)
            anim.save(str(save_path), writer=writer, dpi=dpi)

    return anim
