import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from src.core.evaluator.base_evaluator import Evaluator



class ODEEvaluator(Evaluator):
    def __init__(self, provider, metric, logger, freq=10):
        self.provider = provider
        self.metric = metric
        self.logger = logger
        self.freq = freq

    def run(self, trainer):
        if trainer.epoch % self.freq != 0:
            return

        x, y_true = self.provider.get()

        with torch.no_grad():
            y_pred = trainer.model(x.to(trainer.device))
            loss = self.metric(y_pred, y_true.to(trainer.device))

        self.logger.add_scalar("Evaluation/MSE", loss.item(), trainer.epoch)


    def compute_ode_evaluation(
        self, epoch: int, batch: Any = None, prefix: str = "Evaluation"
    ) -> None:
        """Evaluate the model against the numerical reference solution of the ODE.

        Uses the scipy solver (via ``self.ode.simulate``) to obtain the reference
        trajectory, then compares the network predictions to that trajectory on
        the same time grid.

        The following metrics and visualisations are logged to TensorBoard:

        - ``Evaluation/MSE``: Global MSE between predictions and reference solution.
        - ``Evaluation/MSE/<var_name>``: Per-variable MSE (e.g. ``prey``, ``predator``).
        - ``Evaluation/Observables/DynamicTrajectories``: Time-domain trajectories.
        - ``Evaluation/Observables/DynamicPhaseTrajectories``: Phase-space portrait.
        - ``Evaluation/Observables/PhasePortraitOverlay``: Phase portrait accumulated
          across epochs to visualise convergence toward the reference attractor.
        - ``Evaluation/Residuals/CollocationHeatmap``: Heatmap of ODE residuals on
          the collocation grid, one row per state variable.

        The model is restored to ``train()`` mode before returning, even if the
        solver fails to produce a solution.

        Args:
            epoch: Current epoch, used as the x-axis in TensorBoard.
            batch: Data batch (unused here, kept for a consistent signature).
            prefix: Log prefix (not used directly, kept for consistency).
        """
        self.model.eval()

        if self.lambda_ode > 0:
            sol = self.ode.simulate(
                t_span=self.ode_experiment_config.ode_config.t_span,
                x0=self.x0.cpu().numpy(),
                nb_points=self.ode_experiment_config.ode_config.grid_size,
            )

            if len(sol.t) == 0 or len(sol.y) == 0:
                self.logger.info(
                    f"Validation solver returned no solution at epoch {epoch}. "
                    "Skipping evaluation. Check ODE specifications and parameters."
                )
                self.model.train()
                return

            t_true = sol.t
            y_true = sol.y.T  # scipy: (dim, N) → (N, dim)

            t_tensor = torch.tensor(t_true, dtype=torch.float32).unsqueeze(1).to(self.device)
            y_true_tensor = torch.tensor(y_true, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                y_pred = self.model(t_tensor)
                mse = torch.mean((y_pred - y_true_tensor) ** 2)
                self.writer.add_scalar("Evaluation/MSE", mse.item(), epoch)
                self.log_mse_per_variable(
                    y_pred=y_pred,
                    y_true_tensor=y_true_tensor,
                    epoch=epoch,
                )

            y_pred_np = y_pred.cpu().numpy()

            self.log_trajectory_phase_space_plot(
                y_true=y_true,
                y_pred=y_pred_np,
                tensorboard_path="Evaluation/Observables/DynamicPhaseTrajectories",
                epoch_step=epoch,
            )
            self.log_trajectory_plot(
                t_true,
                y_true=y_true,
                y_pred=y_pred_np,
                tensorboard_path="Evaluation/Observables/DynamicTrajectories",
                epoch_step=epoch,
            )

            # Bug 2 fix: physics_loss_residual uses autograd.grad internally
            # and must never be called inside torch.no_grad().
            self.log_collocation_residual_heatmap(epoch=epoch)

            self.log_phase_portrait_overlay(
                y_true=y_true,
                y_pred=y_pred_np,
                epoch=epoch,
            )

        self.model.train()

    def log_collocation_residual_heatmap(self, epoch: int) -> None:
        """Log a heatmap of ODE residuals on the collocation grid to TensorBoard.

        Evaluates the physics residual ``f(t, u, u')`` at every collocation point
        and renders it as a heatmap with time on the x-axis and one row per state
        variable. This reveals temporal regions where the ODE is poorly satisfied,
        which is typically the most actionable diagnostic for a PINN.

        Note:
            ``physics_loss_residual`` calls ``torch.autograd.grad`` internally and
            must not be wrapped in ``torch.no_grad()``. This method is therefore
            called outside any ``no_grad`` context in ``compute_ode_evaluation``.

        The image is logged under ``Evaluation/Residuals/CollocationHeatmap``.

        Args:
            epoch: Current epoch, used as the x-axis in TensorBoard.
        """
        # Bug 2 fix: no torch.no_grad() wrapper here — autograd.grad requires
        # the computation graph to be intact.
        residuals = self.physics_loss_residual()  # (N, n_vars)

        residuals_np = residuals.detach().abs().cpu().numpy().T  # (n_vars, N)
        n_vars = residuals_np.shape[0]
        t_np = self.t.detach().squeeze().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, max(2, n_vars * 1.2)))
        im = ax.imshow(
            residuals_np,
            aspect="auto",
            cmap="hot",
            extent=[t_np.min(), t_np.max(), -0.5, n_vars - 0.5],
            origin="lower",
            interpolation="nearest",
        )
        ax.set_yticks(range(n_vars))
        ax.set_yticklabels(self.var_names)
        ax.set_xlabel("t")
        ax.set_title(f"ODE residuals — epoch {epoch}")
        fig.colorbar(im, ax=ax, label="|residual|")
        fig.tight_layout()

        self.writer.add_image(
            "Evaluation/Residuals/CollocationHeatmap",
            self._fig_to_tensorboard(fig),
            epoch,
        )
        plt.close(fig)

    def log_phase_portrait_overlay(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epoch: int,
    ) -> None:
        """Log a phase portrait overlaying predictions from multiple epochs.

        Accumulates predicted trajectories across evaluation calls and renders
        them on the reference phase portrait. Earlier epochs are drawn in a
        lighter colour; the current epoch is drawn in full opacity. This gives
        an intuitive view of how the network converges toward the reference
        attractor over training.

        Only applicable to systems with at least 2 state variables. For systems
        with more than 2 variables, the first two are used (index 0 and 1).

        The image is logged under ``Evaluation/Observables/PhasePortraitOverlay``.

        Args:
            y_true: Reference trajectory of shape ``(N, n_vars)``.
            y_pred: Predicted trajectory of shape ``(N, n_vars)`` at the current epoch.
            epoch: Current epoch, used as the x-axis in TensorBoard.
        """
        if y_pred.shape[1] < 2:
            return  # Phase portrait requires at least 2 variables.

        self._phase_overlay_history.append((epoch, y_pred))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(y_true[:, 0], y_true[:, 1], "k-", linewidth=2, label="Reference", zorder=10)

        n_history = len(self._phase_overlay_history)
        cmap = plt.cm.Blues

        for idx, (ep, y_hist) in enumerate(self._phase_overlay_history):
            alpha = 0.15 + 0.7 * (idx / max(n_history - 1, 1))
            color = cmap(0.3 + 0.6 * (idx / max(n_history - 1, 1)))
            label = f"epoch {ep}" if idx == n_history - 1 else None
            ax.plot(y_hist[:, 0], y_hist[:, 1], color=color, alpha=alpha,
                    linewidth=1.2, label=label)

        ax.set_xlabel(self.var_names[0])
        ax.set_ylabel(self.var_names[1])
        ax.set_title(f"Phase portrait — epoch {epoch}")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        self.writer.add_image(
            "Evaluation/Observables/PhasePortraitOverlay",
            self._fig_to_tensorboard(fig),
            epoch,
        )
        plt.close(fig)

    def log_trajectory_plot(
        self,
        t_true: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epoch_step: int,
        tensorboard_path: str,
    ) -> None:
        """Log a time-domain trajectory plot to TensorBoard.

        Delegates figure construction to ``self.ode.log_trajectory_plot``.
        If that method returns ``None`` (e.g. the ODE has no plot implementation),
        no image is logged.

        Args:
            t_true: Reference time vector of shape ``(N,)``.
            y_true: Reference trajectory of shape ``(N, model_dimension)``.
            y_pred: Predicted trajectory of shape ``(N, model_dimension)``.
            epoch_step: Current epoch, used as the x-axis in TensorBoard.
            tensorboard_path: TensorBoard tag under which the image is logged.
        """
        img = self.ode.log_trajectory_plot(t_true=t_true, y_true=y_true, y_pred=y_pred)
        if img is not None:
            self.writer.add_image(tensorboard_path, img, epoch_step)

    def log_trajectory_phase_space_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epoch_step: int,
        tensorboard_path: str,
    ) -> None:
        """Log a phase-space portrait to TensorBoard.

        Delegates figure construction to ``self.ode.log_trajectory_phase_space_plot``.
        If that method returns ``None``, no image is logged.

        Args:
            y_true: Reference trajectory of shape ``(N, model_dimension)``.
            y_pred: Predicted trajectory of shape ``(N, model_dimension)``.
            epoch_step: Current epoch, used as the x-axis in TensorBoard.
            tensorboard_path: TensorBoard tag under which the image is logged.
        """
        img = self.ode.log_trajectory_phase_space_plot(y_true=y_true, y_pred=y_pred)
        if img is not None:
            self.writer.add_image(tensorboard_path, img, epoch_step)

    @staticmethod
    def _fig_to_tensorboard(fig: plt.Figure) -> torch.Tensor:
        """Convert a Matplotlib figure to a CHW uint8 tensor for TensorBoard.

        Reads the RGB buffer directly from the Matplotlib canvas without any
        intermediate file or external dependency.

        Args:
            fig: A Matplotlib figure to convert.

        Returns:
            A tensor of shape ``(3, H, W)`` with dtype ``uint8``, ready to be
            passed to ``SummaryWriter.add_image``.
        """
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        img = img[:, :, 1:].copy()  # ARGB → RGB + rendre le tableau writable

        return torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW