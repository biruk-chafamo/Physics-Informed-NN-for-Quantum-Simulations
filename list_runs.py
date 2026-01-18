#!/usr/bin/env python3
"""
Command-line interface for listing and searching saved PINN runs.

Usage:
    python list_runs.py                              # List all runs
    python list_runs.py --potential gaussian         # Filter by potential
    python list_runs.py --details 20240115_143022_gaussian_h32x32x32
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from pinn_schrodinger.run_manager import RunManager


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='List and search saved PINN training runs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Filters
    parser.add_argument(
        '--potential',
        type=str,
        choices=['gaussian', 'harmonic', 'infinite_well'],
        help='Filter by potential type'
    )
    parser.add_argument(
        '--tags',
        type=str,
        nargs='*',
        help='Filter by tags (runs must have all specified tags)'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=None,
        help='Maximum number of runs to show'
    )

    # Output options
    parser.add_argument(
        '--details',
        type=str,
        metavar='RUN_ID',
        help='Show detailed information for a specific run'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output in JSON format'
    )

    return parser.parse_args()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_datetime(iso_string: str) -> str:
    """Format ISO datetime string."""
    if not iso_string:
        return "N/A"
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_string[:16] if len(iso_string) > 16 else iso_string


def print_run_summary(run: dict) -> None:
    """Print a single run in summary format."""
    run_id = run.get('run_id', 'unknown')
    created = format_datetime(run.get('created_at', ''))
    potential = run.get('config', {}).get('potential', {}).get('type', 'N/A')
    hidden = run.get('config', {}).get('model', {}).get('hidden_dims', [])
    hidden_str = 'x'.join(str(d) for d in hidden) if hidden else 'N/A'
    final_loss = run.get('final_loss')
    loss_str = f"{final_loss:.4f}" if final_loss else "N/A"
    training_time = run.get('training_time_seconds')
    time_str = format_time(training_time) if training_time else "N/A"
    epochs = run.get('epochs_trained', 'N/A')
    tags = run.get('tags', [])
    tags_str = ', '.join(tags) if tags else ''

    print(f"  {run_id}")
    print(f"    Created: {created}  Potential: {potential}  Arch: h{hidden_str}")
    print(f"    Loss: {loss_str}  Time: {time_str}  Epochs: {epochs}")
    if tags_str:
        print(f"    Tags: {tags_str}")
    print()


def print_run_details(run_id: str, metadata: dict) -> None:
    """Print detailed information for a run."""
    print("=" * 60)
    print(f"Run: {run_id}")
    print("=" * 60)
    print()

    # Timing
    print("Timing:")
    print(f"  Created: {format_datetime(metadata.get('created_at', ''))}")
    print(f"  Training started: {format_datetime(metadata.get('training_start', ''))}")
    print(f"  Training ended: {format_datetime(metadata.get('training_end', ''))}")
    training_time = metadata.get('training_time_seconds')
    if training_time:
        print(f"  Training duration: {format_time(training_time)} ({training_time:.2f}s)")
    print()

    # Model
    print("Model:")
    config = metadata.get('config', {})
    model_config = config.get('model', {})
    print(f"  Hidden dims: {model_config.get('hidden_dims', 'N/A')}")
    print(f"  Activation: {model_config.get('activation', 'N/A')}")
    print(f"  Parameters: {metadata.get('model_parameters', 'N/A'):,}")
    print()

    # Training
    print("Training:")
    training_config = config.get('training', {})
    print(f"  Epochs: {metadata.get('epochs_trained', 'N/A')}")
    print(f"  Learning rate: {training_config.get('lr', 'N/A')}")
    print(f"  Physics weight: {training_config.get('physics_weight', 'N/A')}")
    print(f"  Initial weight: {training_config.get('initial_weight', 'N/A')}")
    print(f"  Boundary weight: {training_config.get('boundary_weight', 'N/A')}")
    print(f"  Normalization weight: {training_config.get('normalization_weight', 'N/A')}")
    print()

    # Loss
    print("Final Loss:")
    print(f"  Total: {metadata.get('final_loss', 'N/A')}")
    loss_components = metadata.get('final_loss_components', {})
    if loss_components:
        print(f"  Physics: {loss_components.get('physics', 'N/A')}")
        print(f"  Initial: {loss_components.get('initial', 'N/A')}")
        print(f"  Boundary: {loss_components.get('boundary', 'N/A')}")
        print(f"  Normalization: {loss_components.get('normalization', 'N/A')}")
    print()

    # Domain
    print("Domain:")
    domain_config = config.get('domain', {})
    print(f"  x: [{domain_config.get('x_min', 'N/A')}, {domain_config.get('x_max', 'N/A')}]")
    print(f"  t: [{domain_config.get('t_min', 'N/A')}, {domain_config.get('t_max', 'N/A')}]")
    print(f"  Grid: {domain_config.get('nx', 'N/A')} x {domain_config.get('nt', 'N/A')}")
    print()

    # Potential
    print("Potential:")
    potential_config = config.get('potential', {})
    print(f"  Type: {potential_config.get('type', 'N/A')}")
    print(f"  Amplitude: {potential_config.get('amplitude', 'N/A')}")
    print(f"  Center: {potential_config.get('center', 'N/A')}")
    print(f"  Sigma: {potential_config.get('sigma', 'N/A')}")
    print()

    # Initial condition
    print("Initial Condition:")
    ic_config = config.get('initial_condition', {})
    print(f"  Type: {ic_config.get('type', 'N/A')}")
    print(f"  Amplitude: {ic_config.get('amplitude', 'N/A')}")
    print(f"  Center: {ic_config.get('center', 'N/A')}")
    print(f"  Sigma: {ic_config.get('sigma', 'N/A')}")
    print()

    # Tags
    tags = metadata.get('tags', [])
    if tags:
        print(f"Tags: {', '.join(tags)}")
        print()

    # Environment
    env = metadata.get('environment', {})
    if env:
        print("Environment:")
        print(f"  Platform: {env.get('platform', 'N/A')}")
        print(f"  Python: {env.get('python_version', 'N/A').split()[0]}")
        print(f"  PyTorch: {env.get('pytorch_version', 'N/A')}")
        print()

    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    run_manager = RunManager("runs")

    # Check if runs directory exists
    if not Path("runs").exists():
        print("No runs directory found at 'runs'")
        print("Run 'python train.py' to create your first run.")
        return 0

    # Show details for specific run
    if args.details:
        try:
            metadata = run_manager.get_run_metadata(args.details)
            if args.json:
                print(json.dumps(metadata, indent=2))
            else:
                print_run_details(args.details, metadata)
        except FileNotFoundError:
            print(f"Run not found: {args.details}")
            return 1
        return 0

    # List runs
    runs = run_manager.list_runs(
        potential=args.potential,
        tags=args.tags,
        limit=args.limit,
    )

    if not runs:
        if args.potential or args.tags:
            print("No runs found matching filters.")
        else:
            print("No runs found.")
            print("Run 'python train.py' to create your first run.")
        return 0

    if args.json:
        print(json.dumps(runs, indent=2))
        return 0

    # Print header
    filters = []
    if args.potential:
        filters.append(f"potential={args.potential}")
    if args.tags:
        filters.append(f"tags={','.join(args.tags)}")

    print("=" * 60)
    print(f"Saved Runs ({len(runs)} found)")
    if filters:
        print(f"Filters: {', '.join(filters)}")
    print("=" * 60)
    print()

    for run in runs:
        print_run_summary(run)

    print("-" * 60)
    print("Usage:")
    print("  python list_runs.py --details <run_id>    # Show full details")
    print("  python infer.py --run-id <run_id> --plot  # Run inference")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
