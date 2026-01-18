#!/usr/bin/env python3
"""
Command-line interface for training the PINN Schrödinger solver.

Usage:
    python train.py                              # Use default config
    python train.py --config configs/custom.yaml # Use custom config
    python train.py --epochs 5000 --lr 0.0001    # Override specific params
    python train.py --potential harmonic         # Use different potential
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from pinn_schrodinger.config import Config, load_config, DomainConfig, ModelConfig, TrainingConfig, PotentialConfig, InitialConditionConfig
from pinn_schrodinger.model import SchrodingerPINN
from pinn_schrodinger.potentials import create_potential_from_config
from pinn_schrodinger.physics import create_grid, create_initial_condition
from pinn_schrodinger.trainer import SchrodingerTrainer
from pinn_schrodinger.visualization import (
    plot_initial_condition,
    plot_probability_density,
    plot_loss_curves,
    plot_boundary_check,
    create_animation,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a PINN to solve the 1D Schrödinger equation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML configuration file'
    )

    # Domain parameters
    parser.add_argument('--nx', type=int, help='Number of spatial points')
    parser.add_argument('--nt', type=int, help='Number of temporal points')
    parser.add_argument('--x-max', type=float, help='Right spatial boundary')
    parser.add_argument('--t-max', type=float, help='Final time')

    # Model parameters
    parser.add_argument(
        '--hidden-dims',
        type=int,
        nargs='+',
        help='Hidden layer dimensions (e.g., --hidden-dims 32 32 32)'
    )
    parser.add_argument(
        '--activation',
        type=str,
        choices=['leaky_relu', 'relu', 'tanh', 'silu', 'gelu'],
        help='Activation function'
    )

    # Training parameters
    parser.add_argument('--epochs', '-e', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--log-every', type=int, help='Logging frequency (epochs)')

    # Potential parameters
    parser.add_argument(
        '--potential',
        type=str,
        choices=['gaussian', 'harmonic', 'infinite_well'],
        help='Potential type'
    )
    parser.add_argument('--potential-amplitude', type=float, help='Potential amplitude')
    parser.add_argument('--potential-center', type=float, help='Potential center')
    parser.add_argument('--potential-sigma', type=float, help='Potential width (Gaussian)')

    # Output
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output',
        help='Directory for output files'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save trained model to output directory'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    parser.add_argument(
        '--animate',
        action='store_true',
        help='Create animated GIF of wave function evolution'
    )
    parser.add_argument(
        '--animation-frames',
        type=int,
        default=100,
        help='Number of frames in animation'
    )
    parser.add_argument(
        '--animation-fps',
        type=int,
        default=20,
        help='Frames per second for animation'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    return parser.parse_args()


def build_config(args) -> Config:
    """Build configuration from args, optionally loading from YAML first."""
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()

    # Override domain config
    if args.nx is not None:
        config.domain.nx = args.nx
    if args.nt is not None:
        config.domain.nt = args.nt
    if args.x_max is not None:
        config.domain.x_max = args.x_max
    if args.t_max is not None:
        config.domain.t_max = args.t_max

    # Override model config
    if args.hidden_dims is not None:
        config.model.hidden_dims = args.hidden_dims
    if args.activation is not None:
        config.model.activation = args.activation

    # Override training config
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.lr = args.lr
    if args.log_every is not None:
        config.training.log_every = args.log_every

    # Override potential config
    if args.potential is not None:
        config.potential.type = args.potential
    if args.potential_amplitude is not None:
        config.potential.amplitude = args.potential_amplitude
    if args.potential_center is not None:
        config.potential.center = args.potential_center
    if args.potential_sigma is not None:
        config.potential.sigma = args.potential_sigma

    return config


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Build configuration
    config = build_config(args)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    if not args.quiet:
        print("=" * 60)
        print("PINN Schrödinger Solver")
        print("=" * 60)
        print(f"Domain: x=[{config.domain.x_min}, {config.domain.x_max}], "
              f"t=[{config.domain.t_min}, {config.domain.t_max}]")
        print(f"Grid: {config.domain.nx} x {config.domain.nt}")
        print(f"Model: hidden_dims={config.model.hidden_dims}, "
              f"activation={config.model.activation}")
        print(f"Training: epochs={config.training.epochs}, lr={config.training.lr}")
        print(f"Potential: {config.potential.type}")
        print(f"Output: {output_dir}")
        print("=" * 60)
        print()

    # Create components
    logger.info("Initializing model and trainer...")
    model = SchrodingerPINN(config.model)
    potential = create_potential_from_config(config.potential)
    trainer = SchrodingerTrainer(model, potential, config)

    if not args.quiet:
        print(model)
        print(f"Potential: {potential}")
        print()

    # Create grid and initial condition
    x, t, x_t = create_grid(config.domain)
    phi_0 = create_initial_condition(
        x,
        amplitude=config.initial_condition.amplitude,
        center=config.initial_condition.center,
        sigma=config.initial_condition.sigma,
        dx=config.domain.dx
    )

    # Plot initial setup
    if not args.no_plots:
        fig = plot_initial_condition(x, phi_0, potential,
                                     save_path=output_dir / 'initial_condition.png')
        plt.close(fig)
        logger.info(f"Saved initial condition plot to {output_dir / 'initial_condition.png'}")

    # Train
    logger.info("Starting training...")
    result = trainer.train(x_t, phi_0, verbose=not args.quiet)
    logger.info("Training complete!")

    # Save model
    if args.save_model:
        model_path = output_dir / 'model.pt'
        model.save(str(model_path))
        logger.info(f"Saved model to {model_path}")

    # Generate plots
    if not args.no_plots:
        # Loss curves
        fig = plot_loss_curves(
            result.losses,
            result.loss_components,
            result.epochs_logged,
            save_path=output_dir / 'loss_curves.png'
        )
        plt.close(fig)
        logger.info(f"Saved loss curves to {output_dir / 'loss_curves.png'}")

        # Probability density
        if result.predictions:
            final_pred = result.predictions[-1]
            fig = plot_probability_density(
                x, t, final_pred, phi_0, config.domain,
                num_snapshots=10,
                save_path=output_dir / 'probability_density.png'
            )
            plt.close(fig)
            logger.info(f"Saved probability density to {output_dir / 'probability_density.png'}")

            # Boundary check
            fig = plot_boundary_check(
                t, final_pred, config.domain,
                save_path=output_dir / 'boundary_check.png'
            )
            plt.close(fig)
            logger.info(f"Saved boundary check to {output_dir / 'boundary_check.png'}")

            # Animation
            if args.animate:
                if not args.quiet:
                    print("Creating animation...")
                anim = create_animation(
                    x, t, final_pred, phi_0,
                    domain=config.domain,
                    num_frames=args.animation_frames,
                    fps=args.animation_fps,
                    save_path=output_dir / 'wave_evolution.gif'
                )
                plt.close()
                logger.info(f"Saved animation to {output_dir / 'wave_evolution.gif'}")
                if not args.quiet:
                    print(f"Animation saved to {output_dir / 'wave_evolution.gif'}")

    # Print summary
    if not args.quiet:
        print()
        print("=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Final loss: {result.losses[-1]:.4f}")
        if result.loss_components:
            final_lc = result.loss_components[-1]
            print(f"  Physics:       {final_lc['physics']:.4f}")
            print(f"  Initial cond:  {final_lc['initial']:.4f}")
            print(f"  Boundary cond: {final_lc['boundary']:.4f}")
            print(f"  Normalization: {final_lc['normalization']:.4f}")
        print(f"Output saved to: {output_dir}")
        print("=" * 60)


if __name__ == '__main__':
    main()
