import torch
from src.core.registry import REGISTRY
from src.repositories.losses import AvailablesLoss

@REGISTRY.losses.register(AvailablesLoss.PINN_LOSS)
class PINNLoss:
    """
    PINN loss = lambda_ode * L_physics + lambda_data * L_data

    Args:
        ode:          ODE object exposing a ``torch_ode(y) -> dy/dt`` method.
        lambda_ode:   Weight for the physics residual loss.
        lambda_data:  Weight for the supervised data loss.
    """

    def __init__(self, ode=None, lambda_ode: float = 1.0, lambda_data: float = 1.0):
        self.ode = ode
        self.lambda_ode = lambda_ode
        self.lambda_data = lambda_data

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_derivative(self, y_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute dy/dt via autograd.

        Args:
            y_pred: Network output of shape (N, n_vars).
            t:      Collocation points of shape (N, 1), requires_grad=True.

        Returns:
            Gradient tensor of shape (N, n_vars).
        """
        return torch.autograd.grad(
            outputs=y_pred,
            inputs=t,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

    def _physics_loss(self, model: torch.nn.Module, t: torch.Tensor) -> torch.Tensor:
        """
        ODE residual loss: mean((dy/dt - F(y, t))²)

        Note:
            Must NOT be called inside torch.no_grad().

        Args:
            model: PINN model.
            t:     Collocation points, requires_grad=True.

        Returns:
            Scalar loss tensor.
        """
        y_pred = model(t)
        dy_dt = self._compute_derivative(y_pred, t)
        residuals = (dy_dt - self.ode.torch_ode(y_pred)) ** 2
        return residuals.mean()

    def _data_loss(self, model: torch.nn.Module, batch: dict) -> torch.Tensor:
        """
        Supervised MSE loss: mean((y_pred - y_obs)²)

        Args:
            model: PINN model.
            batch: Dict with keys "x" (N, input_dim) and "y" (N, output_dim).

        Returns:
            Scalar loss tensor.
        """
        x, y_obs = batch["x"], batch["y"]
        y_pred = model(x)
        return ((y_pred - y_obs) ** 2).mean()

    # ── Public interface ──────────────────────────────────────────────────────

    def __call__(
        self,
        model: torch.nn.Module,
        batch: dict,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        """
        Compute total loss and return a detailed breakdown.

        Args:
            model: PINN model.
            batch: Dict with keys "x" and "y".
            t:     Collocation points, requires_grad=True.

        Returns:
            Dict with keys:
                - "total"   : weighted sum (always present)
                - "physics" : physics residual loss (None if lambda_ode == 0)
                - "data"    : supervised data loss  (None if lambda_data == 0)
        """
        total = torch.tensor(0.0)
        physics_loss = None
        data_loss = None

        if self.lambda_ode > 0 and self.ode is not None:
            physics_loss = self._physics_loss(model, t)
            total = total + self.lambda_ode * physics_loss

        if self.lambda_data > 0 and batch is not None:
            data_loss = self._data_loss(model, batch)
            total = total + self.lambda_data * data_loss

        return {
            "total":   total,
            "physics": physics_loss,
            "data":    data_loss,
        }