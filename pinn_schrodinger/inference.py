"""
Inference utilities for the PINN Schrödinger solver.

Provides a high-level interface for loading trained models and running predictions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from .config import Config
from .model import SchrodingerPINN
from .run_manager import RunManager


@dataclass
class PredictionResult:
    """Container for prediction results.

    Attributes:
        x: Spatial coordinates (nx,).
        t: Temporal coordinates (nt,).
        x_t: Grid coordinates (nx*nt, 2).
        psi: Complex wave function (nx*nt,).
        psi_real: Real part of wave function (nx*nt,).
        psi_imag: Imaginary part of wave function (nx*nt,).
        probability: Probability density |ψ|² (nx*nt,).
    """
    x: torch.Tensor
    t: torch.Tensor
    x_t: torch.Tensor
    psi: torch.Tensor
    psi_real: torch.Tensor
    psi_imag: torch.Tensor
    probability: torch.Tensor


class PINNPredictor:
    """High-level interface for running inference with trained PINN models.

    Example usage:
        # Load from a run
        predictor = PINNPredictor(run_id="20240115_143022_gaussian_h32x32x32")

        # Predict on training domain
        result = predictor.predict()

        # Extrapolate to longer time
        result = predictor.predict_extrapolated(t_max_new=10.0)

        # Get snapshot at specific time
        x, psi = predictor.predict_at_time(t_value=2.5)
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        model_path: Optional[str] = None,
        runs_dir: str = "runs",
        device: Optional[torch.device] = None,
    ):
        """Initialize the predictor by loading a trained model.

        Args:
            run_id: Run ID to load from (uses RunManager).
            model_path: Direct path to model.pt file.
            runs_dir: Directory containing runs (default: "runs").
            device: Device to run predictions on.

        Raises:
            ValueError: If neither run_id nor model_path is provided.
            FileNotFoundError: If run or model file doesn't exist.
        """
        if run_id is None and model_path is None:
            raise ValueError("Must provide either run_id or model_path")

        self.device = device or torch.device("cpu")
        self.run_id = run_id
        self.metadata = None

        if run_id is not None:
            # Load from run
            self.run_manager = RunManager(runs_dir)
            self.model, self.config, self.metadata = self.run_manager.load_run(
                run_id, device=self.device
            )
        else:
            # Load from direct path
            self.run_manager = None
            self.model, self.config = SchrodingerPINN.load_with_config(
                model_path, device=self.device
            )

            if self.config is None:
                # Fall back to default config if not saved with model
                self.config = Config()

        self.model.eval()

    def predict(
        self,
        x: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
    ) -> PredictionResult:
        """Run prediction on given coordinates.

        Args:
            x: Spatial coordinates. If None, uses config domain.
            t: Temporal coordinates. If None, uses config domain.
            x_t: Pre-constructed grid (overrides x and t if provided).

        Returns:
            PredictionResult with wave function values.
        """
        # Create coordinates if not provided
        if x_t is None:
            if x is None:
                x = torch.linspace(
                    self.config.domain.x_min,
                    self.config.domain.x_max,
                    self.config.domain.nx,
                    device=self.device,
                )
            if t is None:
                t = torch.linspace(
                    self.config.domain.t_min,
                    self.config.domain.t_max,
                    self.config.domain.nt,
                    device=self.device,
                )

            # Create meshgrid
            x_grid, t_grid = torch.meshgrid(x, t, indexing="ij")
            x_t = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)

        x_t = x_t.to(self.device)

        # Run prediction
        with torch.no_grad():
            output = self.model(x_t)
            psi_real = output[:, 0]
            psi_imag = output[:, 1]
            psi = psi_real + 1j * psi_imag
            probability = torch.abs(psi) ** 2

        # Extract x and t from x_t if not provided
        if x is None:
            x = torch.unique(x_t[:, 0])
        if t is None:
            t = torch.unique(x_t[:, 1])

        return PredictionResult(
            x=x,
            t=t,
            x_t=x_t,
            psi=psi,
            psi_real=psi_real,
            psi_imag=psi_imag,
            probability=probability,
        )

    def predict_at_time(
        self,
        t_value: float,
        x: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a snapshot of the wave function at a specific time.

        Args:
            t_value: Time value to evaluate at.
            x: Spatial coordinates. If None, uses config domain.

        Returns:
            Tuple of (x, psi) where psi is the complex wave function.
        """
        if x is None:
            x = torch.linspace(
                self.config.domain.x_min,
                self.config.domain.x_max,
                self.config.domain.nx,
                device=self.device,
            )

        # Create x_t tensor for single time
        t_tensor = torch.full((len(x),), t_value, device=self.device)
        x_t = torch.stack([x, t_tensor], dim=1)

        with torch.no_grad():
            psi = self.model.predict_complex(x_t)

        return x, psi

    def predict_extrapolated(
        self,
        t_max_new: float,
        nt_new: Optional[int] = None,
        x: Optional[torch.Tensor] = None,
    ) -> PredictionResult:
        """Extrapolate predictions beyond the training domain.

        Args:
            t_max_new: New maximum time (can be > training t_max).
            nt_new: Number of time points. If None, uses config nt.
            x: Spatial coordinates. If None, uses config domain.

        Returns:
            PredictionResult with extrapolated wave function values.
        """
        if x is None:
            x = torch.linspace(
                self.config.domain.x_min,
                self.config.domain.x_max,
                self.config.domain.nx,
                device=self.device,
            )

        if nt_new is None:
            nt_new = self.config.domain.nt

        t = torch.linspace(
            self.config.domain.t_min,
            t_max_new,
            nt_new,
            device=self.device,
        )

        return self.predict(x=x, t=t)

    def get_training_domain(self) -> dict:
        """Get the domain used during training.

        Returns:
            Dictionary with domain bounds.
        """
        return {
            "x_min": self.config.domain.x_min,
            "x_max": self.config.domain.x_max,
            "t_min": self.config.domain.t_min,
            "t_max": self.config.domain.t_max,
            "nx": self.config.domain.nx,
            "nt": self.config.domain.nt,
        }

    def get_model_info(self) -> dict:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        info = {
            "hidden_dims": self.config.model.hidden_dims,
            "activation": self.config.model.activation,
            "parameters": self.model.count_parameters(),
        }

        if self.metadata:
            info.update({
                "run_id": self.run_id,
                "training_time": self.metadata.get("training_time_seconds"),
                "final_loss": self.metadata.get("final_loss"),
                "epochs_trained": self.metadata.get("epochs_trained"),
            })

        return info
