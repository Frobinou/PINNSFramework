from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TrainingStepLog:
    """Snapshot of all training metrics at a given step.

    This dataclass is the single data contract between `train()` and
    `log_training_step()`. It is created once per logging event and
    discarded after writing to TensorBoard.

    Attributes:
        step: Global training step (epoch index).
        total_loss: Combined weighted loss scalar.
        ode_loss: Mean ODE residual across all points and variables.
            `None` when `lambda_ode == 0`.
        data_loss: Mean data loss across the batch.
            `None` when `lambda_data == 0`.
        ode_residuals_per_var: Mean absolute residual per ODE variable,
            computed along the collocation-point axis (dim=0).
            Length equals the number of ODE state variables (e.g. 2 for
            Lotka-Volterra). `None` when `lambda_ode == 0`.
        ode_residuals: Raw residual tensor of shape `(n_points, n_vars)`.
            Kept detached for max/std diagnostics. `None` when
            `lambda_ode == 0`.
    """

    step: int
    total_loss: float
    ode_loss: Optional[float] = None
    data_loss: Optional[float] = None
    ode_residuals_per_var: Optional[list[float]] = None
    ode_residuals: Optional[torch.Tensor] = None

    @classmethod
    def from_tensors(
        cls,
        step: int,
        total_loss: torch.Tensor,
        ode_residuals: Optional[torch.Tensor] = None,
        data_loss: Optional[torch.Tensor] = None,
    ) -> "TrainingStepLog":
        """Build a `TrainingStepLog` from raw training tensors.

        Detaches and converts tensors to Python scalars so the log object
        is safe to pass around without holding onto the computation graph.

        Args:
            step: Current epoch/global step index.
            total_loss: Scalar tensor holding the combined loss.
            ode_residuals: Residual tensor of shape `(n_points, n_vars)`.
                Each column corresponds to one ODE state variable.
                Pass `None` when physics loss is disabled.
            data_loss: Per-sample data loss tensor of shape `(batch,)`.
                Pass `None` when data loss is disabled.

        Returns:
            A fully populated `TrainingStepLog` instance with all tensor
            fields detached and converted to Python scalars.
        """
        per_var = None
        if ode_residuals is not None:
            # Mean absolute residual per variable → one scalar per column.
            per_var = ode_residuals.abs().mean(dim=0).tolist()

        return cls(
            step=step,
            total_loss=total_loss.item(),
            ode_loss=ode_residuals.mean().item() if ode_residuals is not None else None,
            data_loss=data_loss.mean().item() if data_loss is not None else None,
            ode_residuals_per_var=per_var,
            ode_residuals=ode_residuals.detach() if ode_residuals is not None else None,
        )