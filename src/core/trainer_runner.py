import io
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.data_models import TrainingConfig, DataConfig, ODEExperiment
from src.logger import setup_logger
from src.core.checkpoint_manager import CheckpointManager
from src.core.data_loader import ParquetDataLoader
from src.core.models import TrainingStepLog
from src.ai_model_repository import AIMODEL_REPOSITORY
from src.odes import ODE_REPOSITORY


class Trainer:
    """Physics-Informed Neural Network (PINN) trainer.

    Combines a physics loss (ODE residual + initial conditions) with an empirical
    data loss to train a neural network to reproduce the dynamics of a differential
    system.

    The total loss is defined as:

        L = λ_ode * (L_ode + L_ic) + λ_data * L_data

    where:

        - L_ode  : mean ODE residual evaluated on the time grid ``self.t``
        - L_ic   : mean squared error on the initial condition at t=0
        - L_data : mean squared error on empirical data (Parquet)

    Scalars, images, and checkpoints are written to a timestamped directory
    created under ``output_folder_path``.

    Attributes:
        device (str): PyTorch device in use (``"cuda"`` or ``"cpu"``).
        model (torch.nn.Module): Neural network instantiated from ``AIMODEL_REPOSITORY``.
        optimizer (torch.optim.Optimizer): Adam optimizer.
        t (torch.Tensor): Time grid of shape ``(grid_size, 1)``, with gradient enabled.
        x0 (torch.Tensor): Initial conditions of shape ``(model_dimension,)``.
        lambda_ode (float): Weight of the physics loss (0 if no ODE is configured).
        lambda_data (float): Weight of the data loss (0 if no data is configured).
        var_names (list[str]): State variable names used as TensorBoard tag suffixes.
        epoch_step (int): Epoch counter incremented at each call to ``train()``.
        writer (SummaryWriter): TensorBoard SummaryWriter.
        checkpoint_manager (CheckpointManager): Top-k checkpoint manager.
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        ode_experiment_config: ODEExperiment,
        device: str = "",
        output_folder_path: Path = Path(),
    ):
        """Initialize the Trainer and set up all training components.

        Args:
            training_config: Training hyperparameters (epochs, learning rate,
                log frequency, top-k checkpoints, model name).
            ode_experiment_config: ODE experiment configuration (ODE config,
                data config, model dimensions, initial conditions).
            device: Target device (``"cuda"``, ``"cpu"``). If empty, auto-detected
                via ``torch.cuda.is_available()``.
            output_folder_path: Root directory in which a timestamped subdirectory
                ``experiment_YYYY-MM-DD_HH-MM-SS/`` will be created.
        """
        self.logger = setup_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device found: {self.device}")

        self.ode_experiment_config: ODEExperiment = ode_experiment_config
        self.training_config: TrainingConfig = training_config

        self._initialize_training_workspace(output_folder_path)
        self._initialize_training_environment()
        self._initialize_dataloader()
        self._initialize_ode_environment()

    # ---------- Initialization ----------

    def _initialize_dataloader(self) -> None:
        """Build train/val/test dataloaders from Parquet data.

        If ``ode_experiment_config.data_config`` is ``None``, no dataloader is
        created and ``lambda_data`` is forced to ``0.0`` to disable the data loss
        in the training loop.

        Sets:
            self.lambda_data: Weight of the data loss.
            self.dataloader_train: Training DataLoader (or ``None``).
            self.dataloader_val: Validation DataLoader (or ``None``).
            self.dataloader_test: Test DataLoader (or ``None``).
        """
        data_config: Optional[DataConfig] = self.ode_experiment_config.data_config
        if data_config is not None:
            self.logger.info(
                f"Data folder found at {data_config.data_folder}. Initializing dataloader."
            )
            self.lambda_data = data_config.lambda_data
            loader_builder = ParquetDataLoader(
                parquet_path=data_config.data_folder,
                input_cols=self.ode_experiment_config.input_cols,
                target_cols=self.ode_experiment_config.target_cols,
                batch_size=data_config.batch_size
            )
            self.dataloader_train = loader_builder.get_train_loader()
            self.dataloader_val = loader_builder.get_val_loader()
            self.dataloader_test = loader_builder.get_test_loader()
        else:
            self.lambda_data = 0.0
            self.dataloader_train = None
            self.dataloader_val = None
            self.dataloader_test = None

    def _initialize_training_workspace(self, output_folder_path: Path) -> None:
        """Create the experiment directory tree and initialize the TensorBoard writer.

        Directory structure created::

            output_folder_path/
            └── experiment_YYYY-MM-DD_HH-MM-SS/
                ├── tensorboard_logs/
                └── save/

        Args:
            output_folder_path: Parent directory in which the timestamped experiment
                folder will be created.

        Sets:
            self.experiment_folder: Path to the experiment folder.
            self.writer: ``SummaryWriter`` instance pointing to ``tensorboard_logs/``.
            self.save_dir: Checkpoint save directory.
            self.top_k: Maximum number of checkpoints to keep.
            self.log_frequency: Logging and evaluation frequency in epochs.
            self.checkpoint_manager: Top-k checkpoint manager.
            self.epoch_step: Initialized to 0.
        """
        experiment_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_folder = output_folder_path / f"experiment_{experiment_date}"
        self.experiment_folder.mkdir(parents=True, exist_ok=True)

        log_dir = self.experiment_folder / "tensorboard_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        self.save_dir = self.experiment_folder / "save"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.top_k: int = self.training_config.top_k_save_frequency
        self.log_frequency: int = self.training_config.log_frequency

        self.checkpoint_manager = CheckpointManager(self.save_dir, self.top_k, self.logger)
        self.epoch_step = 0

    def _initialize_ode_environment(self) -> None:
        """Instantiate the ODE from the registry and configure its loss weight.

        If ``ode_experiment_config.ode_config`` is ``None``, no ODE is loaded
        and ``lambda_ode`` is forced to ``0.0``.

        ``var_names`` is inferred from ``target_cols`` when available so that all
        TensorBoard tags (residuals, MSE) use meaningful variable labels without
        requiring manual assignment after construction.

        Sets:
            self.ode: ODE instance retrieved from ``ODE_REPOSITORY``.
            self.lambda_ode: Weight of the physics loss.
            self.var_names: State variable names used as TensorBoard tag suffixes.
            self._phase_overlay_history: Accumulator for phase portrait overlay,
                capped at 50 entries to bound memory usage.
        """
        # Bug 4 fix: var_names is now always initialised here, either from
        # target_cols or as generic fallback labels, so no scattered hasattr()
        # checks are needed in logging methods.
        self.var_names: list[str] = (
            list(self.ode_experiment_config.target_cols)
            if self.ode_experiment_config.target_cols
            else [f"var_{i}" for i in range(self.ode_experiment_config.model_dimension)]
        )

        # Bug 3 fix: _phase_overlay_history is initialised here so that
        # log_phase_portrait_overlay can always append to it safely.
        self._phase_overlay_history: deque[tuple[int, np.ndarray]] = deque(maxlen=50)

        ode_config = self.ode_experiment_config.ode_config
        if ode_config is not None:
            self.ode = ODE_REPOSITORY.get(ode_config.ode_name)(params=ode_config.parameters)
            self.lambda_ode = ode_config.lambda_ode
        else:
            self.lambda_ode = 0.0

    def _initialize_training_environment(self) -> None:
        """Instantiate the model, optimizer, time grid, and initial conditions.

        The time grid ``self.t`` spans ``[0, model_dimension]`` with ``grid_size``
        evenly spaced points. ``requires_grad=True`` is enabled to allow automatic
        computation of ``dy/dt`` via ``torch.autograd.grad``.

        Sets:
            self.model: Neural network moved to ``self.device``.
            self.optimizer: Adam optimizer with the configured learning rate.
            self.t: Time grid of shape ``(grid_size, 1)``.
            self.x0: Initial conditions of shape ``(model_dimension,)``.
        """
        self.model = AIMODEL_REPOSITORY.get(self.training_config.model_name)(
            output_dim=self.ode_experiment_config.model_dimension
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.training_config.l_r
        )

        ode_config = self.ode_experiment_config.ode_config

        self.t = torch.linspace(
            0,
            self.ode_experiment_config.model_dimension,
            ode_config.grid_size,
            device=self.device,
        ).view(-1, 1)
        self.t.requires_grad = True

        self.x0 = torch.tensor(
            self.ode_experiment_config.initial_conditions,
            dtype=torch.float32,
            device=self.device,
        )

    # ---------- Core computations ----------

    def compute_derivative(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt via a single autograd pass over all output dimensions.

        Args:
            y_pred: Network output of shape ``(N, n_vars)`` evaluated on ``self.t``.

        Returns:
            Gradient tensor of shape ``(N, n_vars)``.
        """
        grads = torch.autograd.grad(
            outputs=y_pred,
            inputs=self.t,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]
        return grads

    def physics_loss_residual(self) -> torch.Tensor:
        """Compute the ODE residual on the collocation grid.

        Evaluates the gap between the derivative predicted by the network and the
        right-hand side of the ODE:

            residual_i = (dy_i/dt - F_i(y, t))²

        Note:
            This method calls ``torch.autograd.grad`` internally and must **not**
            be wrapped in ``torch.no_grad()``.

        Returns:
            Residual tensor of shape ``(N, n_vars)``, one column per state variable.
        """
        y_pred = self.model(self.t)
        gradient_derivative = self.compute_derivative(y_pred)
        F_t = self.ode.torch_ode(y_pred)
        residuals = (gradient_derivative - F_t) ** 2
        return residuals

    def compute_data_loss(self, batch: dict) -> torch.Tensor:
        """Compute the data residual loss for supervised constraints.

        Args:
            batch: Dictionary with keys:

                - ``"x"`` (Tensor): Input locations of shape ``(batch_size, input_dim)``.
                - ``"y"`` (Tensor): Observed values of shape ``(batch_size, output_dim)``.

        Returns:
            Per-sample squared residual tensor of shape ``(batch_size, output_dim)``.
        """
        x = batch["x"].to(self.device)
        y_obs = batch["y"].to(self.device)
        y_pred = self.model(x)
        residual = (y_pred - y_obs) ** 2
        return residual

    # ---------- Training loop ----------

    def train(self) -> None:
        """Run the full training loop for the configured number of epochs.

        Supports three training modes depending on which lambdas are non-zero:

        - **Physics-only** (``lambda_data == 0``): iterates once per epoch with
          a dummy batch; only the ODE residual loss is computed.
        - **Data-only** (``lambda_ode == 0``): standard mini-batch supervised
          training; no collocation points are evaluated.
        - **Hybrid PINN** (both non-zero): combines data and physics losses
          in each forward pass.

        The TensorBoard logging and checkpoint update are both triggered
        inside this loop via ``log_training_step`` and ``_update_checkpoint``.

        Raises:
            ValueError: If both ``lambda_data`` and ``lambda_ode`` are zero,
                since no loss would be computed.
        """
        if self.lambda_data == 0 and self.lambda_ode == 0:
            raise ValueError("No active loss (lambda_data and lambda_ode are zero).")

        for epoch in range(self.training_config.epochs):
            self.epoch_step += 1
            self.model.train()

            ode_residuals = None
            data_loss = None

            for batch in (iter(self.dataloader_train) if self.lambda_data > 0 else ["dummy_batch"]):
                # "dummy_batch" is used when lambda_data=0 so the loop runs at
                # least once for physics-only training.

                # Bug 6 fix: zero_grad is now inside the batch loop so gradients
                # do not accumulate across batches when the dataloader yields
                # more than one batch per epoch.
                self.optimizer.zero_grad()

                loss = torch.tensor(0.0, device=self.device)

                if self.lambda_ode > 0:
                    ode_residuals = self.physics_loss_residual()
                    # Sum over variables, mean over collocation points.
                    loss = loss + self.lambda_ode * ode_residuals.mean(dim=0).sum()

                if self.lambda_data > 0:
                    data_loss = self.compute_data_loss(batch)
                    loss += self.lambda_data * data_loss.mean()

                loss.backward()
                self.optimizer.step()

            self._update_checkpoint(epoch, loss.item())

            if self.epoch_step % self.log_frequency == 0:
                log = TrainingStepLog.from_tensors(
                    step=self.epoch_step,
                    total_loss=loss,
                    ode_residuals=ode_residuals,
                    data_loss=data_loss,
                )
                self.log_training_step(log)
                self.log_gradients()

    # ---------- Training logging ----------

    def log_training_step(
        self,
        log: TrainingStepLog,
    ) -> None:
        """Write all metrics from a ``TrainingStepLog`` to TensorBoard.

        This is the single dispatch point for all training scalars.
        No ``writer.add_scalar`` calls should be made outside this method.

        TensorBoard tags written:

        - ``Training/loss/total`` — always written.
        - ``Training/loss/physics`` — written when ``log.ode_loss`` is not ``None``.
        - ``Training/loss/data`` — written when ``log.data_loss`` is not ``None``.
        - ``Training/residuals/<var_name>`` — one tag per ODE variable derived from
          ``self.var_names`` (e.g. ``prey``, ``predator`` for Lotka-Volterra).
        - ``Training/residuals/mean`` — global mean residual across all points
          and variables.
        - ``Training/residuals/max`` — maximum absolute residual, useful for
          detecting collocation points where the physics is poorly satisfied.

        Args:
            log: Populated log object produced by ``TrainingStepLog.from_tensors``.
        """
        self.writer.add_scalar("Training/loss/total", log.total_loss, log.step)

        if log.ode_loss is not None:
            self.writer.add_scalar("Training/loss/physics", log.ode_loss, log.step)

        if log.data_loss is not None:
            self.writer.add_scalar("Training/loss/data", log.data_loss, log.step)

        if log.ode_residuals_per_var is not None:
            for name, res in zip(self.var_names, log.ode_residuals_per_var):
                self.writer.add_scalar(f"Training/residuals/{name}", res, log.step)

        if log.ode_residuals is not None:
            self.writer.add_scalar("Training/residuals/mean", log.ode_residuals.mean().item(), log.step)
            self.writer.add_scalar("Training/residuals/max", log.ode_residuals.abs().max().item(), log.step)

    def log_gradients(self) -> None:
        """Compute and log the global gradient L2 norm to TensorBoard.

        Iterates over all model parameters with a gradient, computes the
        per-parameter L2 norm, and aggregates them into a single global norm:

            total_norm = sqrt( sum( ||grad_p||_2^2  for p in parameters ) )

        This metric is useful for diagnosing vanishing or exploding gradients
        during PINN training, where physics and data losses can pull in
        opposite directions.

        The result is written to ``Training/gradients/global_norm``.
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.writer.add_scalar("Training/gradients/global_norm", total_norm, self.epoch_step)


    # ---------- Scalar writer ----------

    def update_scalar_writer(
        self, scalar_dict: dict, epoch: int, prefix: str = "Evaluation"
    ) -> None:
        """Recursively log scalars to TensorBoard.

        Walks ``scalar_dict`` depth-first: values of type ``dict`` trigger a
        recursive call with an extended prefix; scalar values are logged via
        ``self.writer.add_scalar``.

        Example::

            scalar_dict = {
                "total_loss": 0.42,
                "ode": {"combined_ode_loss": 0.3, "ic_loss": 0.1},
                "data": {"data_loss": 0.02},
            }
            # TensorBoard tags generated:
            #   Evaluation/losses/total_loss
            #   Evaluation/losses/ode/combined_ode_loss
            #   Evaluation/losses/ode/ic_loss
            #   Evaluation/losses/data/data_loss

        Args:
            scalar_dict: Flat or nested dictionary of scalar values.
            epoch: Global time step used as the x-axis in TensorBoard.
            prefix: TensorBoard tag prefix, extended at each nesting level.
        """
        for key, value in scalar_dict.items():
            if isinstance(value, dict):
                self.update_scalar_writer(value, epoch, prefix=f"{prefix}/{key}")
            else:
                self.logger.info(f"{prefix}/{key}: {value}")
                self.writer.add_scalar(f"{prefix}/{key}", value, epoch)

    # ---------- Evaluation ----------

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

    def log_mse_per_variable(
        self,
        y_pred: torch.Tensor,
        y_true_tensor: torch.Tensor,
        epoch: int,
    ) -> None:
        """Log per-variable MSE scalars to TensorBoard.

        Decomposes the global MSE into one scalar per state variable so that
        convergence differences between variables are visible in TensorBoard
        (e.g. ``prey`` converging faster than ``predator`` in Lotka-Volterra).

        Tags written: ``Evaluation/MSE/<var_name>`` for each variable.

        Args:
            y_pred: Predicted trajectory tensor of shape ``(N, n_vars)``.
            y_true_tensor: Reference trajectory tensor of shape ``(N, n_vars)``.
            epoch: Current epoch, used as the x-axis in TensorBoard.
        """
        per_var_mse = torch.mean((y_pred - y_true_tensor) ** 2, dim=0)  # (n_vars,)
        for name, mse_val in zip(self.var_names, per_var_mse):
            self.writer.add_scalar(f"Evaluation/MSE/{name}", mse_val.item(), epoch)

    # ---------- Plots ----------

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

    # ---------- Run all ----------

    def run(self) -> None:
        """Orchestrate the full experiment: config, training, and final evaluation.

        Execution sequence:

            1. ``save_config()``            — serialise configs to ``experiment_folder``.
            2. ``train()``                  — run the full training loop.
            3. ``compute_ode_evaluation()`` — final evaluation against the reference solution.
            4. ``writer.close()``           — cleanly close the TensorBoard SummaryWriter.
        """
        self.save_config()
        self.train()
        self.compute_ode_evaluation(epoch=self.epoch_step)
        self.writer.close()

    # ---------- Checkpoint helpers ----------

    def _update_checkpoint(self, epoch: int, loss: float) -> None:
        if (
            len(self.checkpoint_manager.best_checkpoints) < self.training_config.top_k_save_frequency
            or loss < self.checkpoint_manager.best_checkpoints[-1][0]
        ):
            self.checkpoint_manager.save_top_k_checkpoint(
                epoch, loss, self.model, self.optimizer, self.epoch_step
            )

    def save_top_k_checkpoint(self, epoch: int, loss: float) -> None:
        """Save a checkpoint if its loss ranks in the top-k.

        Delegates selection logic and file writing to the ``CheckpointManager``.

        Args:
            epoch: Epoch at which the checkpoint is saved.
            loss: Total loss value at this epoch.
        """
        self.checkpoint_manager.save_top_k_checkpoint(
            epoch, loss, self.model, self.optimizer, self.epoch_step
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load a checkpoint and restore model and optimizer weights.

        Also updates ``self.epoch_step`` with the value stored in the checkpoint.

        Args:
            path: Path to the checkpoint file (``.pt`` or ``.pth``).
        """
        self.epoch_step = self.checkpoint_manager.load_checkpoint(
            path, self.model, self.optimizer
        )

    def save_config(self) -> None:
        """Serialise the experiment configurations to ``experiment_folder``.

        Delegates writing to the ``CheckpointManager``, which persists both
        ``ode_experiment_config`` and ``training_config`` (typically as JSON).
        """
        self.checkpoint_manager.save_config(
            experiment_folder=self.experiment_folder,
            ode_experiment_config=self.ode_experiment_config,
            training_config=self.training_config,
        )