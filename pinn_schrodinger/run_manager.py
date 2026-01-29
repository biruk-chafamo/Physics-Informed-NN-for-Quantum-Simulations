"""
Run management for the PINN Schrödinger solver.

Provides structured saving, loading, and listing of training runs.
"""

import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

from .config import Config
from .model import SchrodingerPINN
from .trainer import TrainingResult


logger = logging.getLogger(__name__)


class RunManager:
    """Manages training runs with structured directory layout.

    Folder structure:
        runs/
            index.json                              # Registry of all runs
            YYYYMMDD_HHMMSS_<potential>_h<dims>/    # Run directory
                metadata.json                       # Full training metadata
                model.pt                            # Model weights + config
                config.yaml                         # Complete config used
                plots/                              # Generated plots
                checkpoints/                        # Intermediate checkpoints
    """

    def __init__(self, runs_dir: str = "runs"):
        """Initialize the run manager.

        Args:
            runs_dir: Base directory for all runs.
        """
        self.runs_dir = Path(runs_dir)
        self.index_path = self.runs_dir / "index.json"

    def generate_run_id(self, config: Config, custom_name: Optional[str] = None) -> str:
        """Generate a run ID from timestamp and config.

        Format: YYYYMMDD_HHMMSS_<potential>_h<hidden_dims>
        Example: 20240115_143022_gaussian_h32x32x32

        Args:
            config: Configuration object.
            custom_name: Optional custom name to use instead of auto-generated.

        Returns:
            Run ID string.
        """
        if custom_name:
            return custom_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        potential_type = config.potential.type
        hidden_dims = "x".join(str(d) for d in config.model.hidden_dims)

        return f"{timestamp}_{potential_type}_h{hidden_dims}"

    def create_run_dir(self, run_id: str) -> Path:
        """Create the directory structure for a run.

        Args:
            run_id: Run identifier.

        Returns:
            Path to the run directory.
        """
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

        return run_dir

    def save_run(
        self,
        run_id: str,
        model: SchrodingerPINN,
        config: Config,
        result: TrainingResult,
        tags: Optional[List[str]] = None,
    ) -> Path:
        """Save a complete training run.

        Args:
            run_id: Run identifier.
            model: Trained model.
            config: Configuration used for training.
            result: Training results.
            tags: Optional list of tags for categorization.

        Returns:
            Path to the run directory.
        """
        run_dir = self.create_run_dir(run_id)

        # Save model with full config
        model_path = run_dir / "model.pt"
        model.save(str(model_path), full_config=config)
        logger.info(f"Saved model to {model_path}")

        # Save config as YAML
        config_path = run_dir / "config.yaml"
        config.save_yaml(str(config_path))
        logger.info(f"Saved config to {config_path}")

        # Build metadata
        final_loss_components = {}
        if result.loss_components:
            final_loss_components = result.loss_components[-1]

        metadata = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "training_start": result.start_time,
            "training_end": result.end_time,
            "training_time_seconds": result.training_time,
            "config": config.to_dict(),
            "final_loss": result.losses[-1] if result.losses else None,
            "final_loss_components": final_loss_components,
            "model_parameters": model.count_parameters(),
            "epochs_trained": len(result.losses),
            "tags": tags or [],
            "environment": {
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "platform": platform.platform(),
            },
        }

        # Add weight history if adaptive weights were used
        if result.weight_history:
            metadata["weight_history"] = result.weight_history

        # Save metadata
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

        # Update index
        self._update_index(run_id, metadata)

        return run_dir

    def load_run(
        self,
        run_id: str,
        device: Optional[torch.device] = None,
    ) -> tuple[SchrodingerPINN, Config, dict]:
        """Load a training run by ID.

        Args:
            run_id: Run identifier.
            device: Device to load the model onto.

        Returns:
            Tuple of (model, config, metadata).

        Raises:
            FileNotFoundError: If run directory doesn't exist.
        """
        run_dir = self.runs_dir / run_id

        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        # Load model with config
        model_path = run_dir / "model.pt"
        model, config = SchrodingerPINN.load_with_config(str(model_path), device=device)

        # Load metadata
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # If config wasn't in the model file, load from YAML
        if config is None:
            from .config import load_config
            config_path = run_dir / "config.yaml"
            config = load_config(str(config_path))

        return model, config, metadata

    def list_runs(
        self,
        potential: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """List all runs, optionally filtered.

        Args:
            potential: Filter by potential type.
            tags: Filter by tags (runs must have all specified tags).
            limit: Maximum number of runs to return.

        Returns:
            List of run metadata dictionaries, sorted by creation time (newest first).
        """
        index = self._load_index()
        runs = list(index.values())

        # Apply filters
        if potential:
            runs = [r for r in runs if r.get("config", {}).get("potential", {}).get("type") == potential]

        if tags:
            runs = [r for r in runs if all(t in r.get("tags", []) for t in tags)]

        # Sort by creation time (newest first)
        runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)

        # Apply limit
        if limit:
            runs = runs[:limit]

        return runs

    def get_run_metadata(self, run_id: str) -> dict:
        """Get metadata for a specific run.

        Args:
            run_id: Run identifier.

        Returns:
            Run metadata dictionary.

        Raises:
            FileNotFoundError: If run doesn't exist.
        """
        run_dir = self.runs_dir / run_id
        metadata_path = run_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        with open(metadata_path, "r") as f:
            return json.load(f)

    def get_run_dir(self, run_id: str) -> Path:
        """Get the directory path for a run.

        Args:
            run_id: Run identifier.

        Returns:
            Path to the run directory.
        """
        return self.runs_dir / run_id

    def run_exists(self, run_id: str) -> bool:
        """Check if a run exists.

        Args:
            run_id: Run identifier.

        Returns:
            True if run exists.
        """
        return (self.runs_dir / run_id).exists()

    def _load_index(self) -> Dict[str, Any]:
        """Load the runs index file.

        Returns:
            Dictionary mapping run IDs to metadata.
        """
        if not self.index_path.exists():
            return {}

        with open(self.index_path, "r") as f:
            return json.load(f)

    def _update_index(self, run_id: str, metadata: dict) -> None:
        """Update the runs index with a new or updated run.

        Args:
            run_id: Run identifier.
            metadata: Run metadata to store in index.
        """
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        index = self._load_index()

        # Store summary info in index (not full config to keep it compact)
        index[run_id] = {
            "run_id": run_id,
            "created_at": metadata.get("created_at"),
            "training_time_seconds": metadata.get("training_time_seconds"),
            "final_loss": metadata.get("final_loss"),
            "model_parameters": metadata.get("model_parameters"),
            "epochs_trained": metadata.get("epochs_trained"),
            "tags": metadata.get("tags", []),
            "config": {
                "potential": metadata.get("config", {}).get("potential", {}),
                "model": metadata.get("config", {}).get("model", {}),
                "domain": metadata.get("config", {}).get("domain", {}),
            },
        }

        with open(self.index_path, "w") as f:
            json.dump(index, f, indent=2)

    def save_checkpoint(
        self,
        run_id: str,
        model: SchrodingerPINN,
        epoch: int,
        config: Optional[Config] = None,
    ) -> Path:
        """Save a training checkpoint.

        Args:
            run_id: Run identifier.
            model: Model to checkpoint.
            epoch: Current epoch number.
            config: Optional config to include.

        Returns:
            Path to the checkpoint file.
        """
        run_dir = self.runs_dir / run_id
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        model.save(str(checkpoint_path), full_config=config)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        return checkpoint_path
