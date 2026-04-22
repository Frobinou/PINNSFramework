import io
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.core.schemas import TrainingConfig, DataConfig, ODEExperiment
from src.logger import setup_logger
from src.core.checkpoint_manager import CheckpointManager
from src.repositories.data_loader.parquet_dataloader import ParquetDataLoader
from src.core.models import TrainingStepLog
from src.repositories.models import AIMODEL_REPOSITORY
from src.repositories.odes import ODE_REPOSITORY


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
