#!/usr/bin/env python3
"""
Command-line interface for running inference with trained PINN models.

Usage:
    python infer.py --run-id 20240115_143022_gaussian_h32x32x32
    python infer.py --run-id my_run --t-max 10.0 --extrapolate --animate
    python infer.py --run-id my_run --time-snapshot 2.5
    python infer.py --run-id my_run --no-plot  # Skip plot generation
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pinn_schrodinger.inference import PINNPredictor
from pinn_schrodinger.visualization import (
    plot_probability_density,
    create_animation,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with a trained PINN model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model loading
    parser.add_argument(
        '--run-id',
        type=str,
        required=True,
        help='Run ID to load (from runs/ directory)'
    )

    # Prediction options
    parser.add_argument(
        '--t-max',
        type=float,
        default=None,
        help='Maximum time for prediction (uses training t_max if omitted)'
    )
    parser.add_argument(
        '--nt',
        type=int,
        default=None,
        help='Number of time points (uses training nt if omitted)'
    )
    parser.add_argument(
        '--extrapolate',
        action='store_true',
        help='Extrapolate beyond training domain (requires --t-max)'
    )
    parser.add_argument(
        '--time-snapshot',
        type=float,
        default=None,
        help='Get wave function at specific time'
    )

    # Output options
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plot generation (plots saved automatically)'
    )
    parser.add_argument(
        '--animate',
        action='store_true',
        help='Create animated GIF of wave function evolution'
    )

    # Other
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output'
    )

    return parser.parse_args()


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

    # Set device
    device = torch.device(args.device)

    # Load model
    if not args.quiet:
        print(f"Loading run: {args.run_id}")

    try:
        predictor = PINNPredictor(
            run_id=args.run_id,
            runs_dir="runs",
            device=device,
        )
    except FileNotFoundError:
        print(f"Error: Run '{args.run_id}' not found in runs/")
        print("Use 'python list_runs.py' to see available runs.")
        return 1

    # Print model info
    if not args.quiet:
        info = predictor.get_model_info()
        domain = predictor.get_training_domain()
        print()
        print("=" * 60)
        print("Model Information")
        print("=" * 60)
        print(f"Hidden dims: {info['hidden_dims']}")
        print(f"Activation: {info['activation']}")
        print(f"Parameters: {info['parameters']:,}")
        if info.get('training_time'):
            print(f"Training time: {info['training_time']:.2f}s")
        if info.get('final_loss'):
            print(f"Final loss: {info['final_loss']:.4f}")
        print()
        print("Training Domain:")
        print(f"  x: [{domain['x_min']}, {domain['x_max']}] ({domain['nx']} points)")
        print(f"  t: [{domain['t_min']}, {domain['t_max']}] ({domain['nt']} points)")
        print("=" * 60)
        print()

    # Run prediction
    if args.time_snapshot is not None:
        # Single time snapshot
        if not args.quiet:
            print(f"Getting snapshot at t={args.time_snapshot}")

        x, psi = predictor.predict_at_time(args.time_snapshot)
        prob = torch.abs(psi) ** 2

        if not args.quiet:
            print(f"x shape: {x.shape}")
            print(f"psi shape: {psi.shape}")
            print(f"Max probability: {prob.max().item():.4f}")

        if not args.no_plot:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            # Wave function
            axes[0].plot(x.cpu().numpy(), psi.real.cpu().numpy(), label='Re(ψ)')
            axes[0].plot(x.cpu().numpy(), psi.imag.cpu().numpy(), label='Im(ψ)')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('ψ')
            axes[0].set_title(f'Wave Function at t={args.time_snapshot}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Probability
            axes[1].plot(x.cpu().numpy(), prob.cpu().numpy())
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('|ψ|²')
            axes[1].set_title('Probability Density')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            save_path = f'inference_snapshot_t{args.time_snapshot}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if not args.quiet:
                print(f"Saved plot to {save_path}")
            plt.close(fig)

    else:
        # Full domain or extrapolated prediction
        if args.extrapolate and args.t_max:
            if not args.quiet:
                print(f"Extrapolating to t_max={args.t_max}")
            result = predictor.predict_extrapolated(
                t_max_new=args.t_max,
                nt_new=args.nt,
            )
        elif args.t_max:
            if not args.quiet:
                print(f"Predicting on domain with t_max={args.t_max}")
            t = torch.linspace(
                predictor.config.domain.t_min,
                args.t_max,
                args.nt or predictor.config.domain.nt,
            )
            result = predictor.predict(t=t)
        else:
            if not args.quiet:
                print("Predicting on training domain")
            result = predictor.predict()

        if not args.quiet:
            print(f"x shape: {result.x.shape}")
            print(f"t shape: {result.t.shape}")
            print(f"psi shape: {result.psi.shape}")
            print(f"Max probability: {result.probability.max().item():.4f}")

        # Generate plots
        if not args.no_plot:
            # Use the same visualization as training
            domain = predictor.config.domain

            # Create a simple domain object for the plot
            class PlotDomain:
                def __init__(self, d):
                    self.nx = d['nx']
                    self.nt = len(result.t)
                    self.x_min = d['x_min']
                    self.x_max = d['x_max']

            plot_domain = PlotDomain(predictor.get_training_domain())

            # Reshape psi for plotting
            psi_grid = result.psi.reshape(len(result.x), len(result.t))

            # Create initial condition (just use first time slice)
            phi_0 = psi_grid[:, 0]

            fig = plot_probability_density(
                result.x,
                result.t,
                result.psi,
                phi_0,
                plot_domain,
                num_snapshots=10,
            )

            save_path = 'inference_probability.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if not args.quiet:
                print(f"Saved plot to {save_path}")
            plt.close(fig)

        # Create animation
        if args.animate:
            if not args.quiet:
                print("Creating animation...")

            domain = predictor.get_training_domain()

            class PlotDomain:
                def __init__(self, d, nt):
                    self.nx = d['nx']
                    self.nt = nt
                    self.x_min = d['x_min']
                    self.x_max = d['x_max']
                    self.t_min = d['t_min']
                    self.t_max = result.t[-1].item()

            plot_domain = PlotDomain(domain, len(result.t))

            psi_grid = result.psi.reshape(len(result.x), len(result.t))
            phi_0 = psi_grid[:, 0]

            save_path = 'inference_animation.gif'

            anim = create_animation(
                result.x,
                result.t,
                result.psi,
                phi_0,
                domain=plot_domain,
                num_frames=100,
                fps=20,
                save_path=save_path,
            )
            plt.close()

            if not args.quiet:
                print(f"Saved animation to {save_path}")

    return 0


if __name__ == '__main__':
    exit(main())
