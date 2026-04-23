import sys
from pathlib import Path

cwd = str(Path.cwd())
sys.path.append(cwd)

import torch


from src.core.schemas import (
    DataConfig,
    TrainingConfig,
    AvailablesODE,
    AvailablesAIModel,
    ODESConfig,
    PhysicsWeights,
    ExperimentConfig
)
from src.repositories.odes.ode_repository.ode_lotka_voltera import ParamsLotkaVoltera
from src.core.callback.tensorboard_callback import TensorBoardCallback
from src.core.callback.checkpoint_callback import CheckpointCallback
from src.core.callback.earlystopping_callback import EarlyStoppingCallback
from src.core.callback.finalevaluation_callback import FinalEvaluationCallback
from src.core.evaluator.mse_evaluator import MSEEvaluator
from src.core.evaluator.ode_evaluator import ODEEvaluator
from src.core.factory import make_dataloader, build_trainer
from src.repositories.losses import AvailablesLoss
from src.logger import setup_logger
from src.core.experiment_io import save_experiment
logger = setup_logger()

# ── Force registration ────────────────────────────────────────────────────────
from src.core.bootstrap import bootstrap_registry
bootstrap_registry()
logger.info("Registries bootstrapped successfully.")

# ── Device & output ───────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
output_folder_path = Path("runs") / "Lotka-Voltera"
logger.info(f"Using device: {device}")

# ── ODE parameters ────────────────────────────────────────────────────────────
ode_params = ParamsLotkaVoltera(alpha=0.67, beta=1.33, delta=1.0, gamma=1.0)

ode_config = ODESConfig(
    parameters=ode_params,
    ode_name=AvailablesODE.LOTKA_VOLTERA,
    grid_size=2000,
    t_span=(0.0, 50.0),
    lambda_ode=1.0,
    initial_conditions=[1.0, 1.0],
    dimension=2,
)
logger.info(f"ODE configuration: {ode_config}")

# ── Data configuration ────────────────────────────────────────────────────────
data_config = DataConfig(
    type="parquet",
    data_path=Path("data/lotka_volterra.parquet"),
    input_cols=["t"],
    target_cols=["prey", "predator"],
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.15,
)

# ── Training configuration ────────────────────────────────────────────────────
training_config = TrainingConfig(
    l_r=1e-3,
    epochs=20,
    top_k_save_frequency=5,
    log_frequency=50,
    model_name=AvailablesAIModel.BASIC_PINN,
    optimizer="Adam",
)

# ── Training configuration ────────────────────────────────────────────────────
physics_weights = PhysicsWeights(
    name=AvailablesLoss.PINN_LOSS,
    lambda_ode=1.0,
    lambda_data=1.0,
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = build_trainer(    
    ode_config=ode_config, 
    loss_config=physics_weights,
    training_config=training_config,
    device=device
)   

experiment_path = save_experiment(ExperimentConfig(ode=ode_config,
                                 data=data_config,
                                 physics_weights=physics_weights,
                                 training=training_config,
                                 device=device),base_dir=output_folder_path)

# ── Callbacks ─────────────────────────────────────────────────────────────────
from src.core.checkpoint_manager import CheckpointManager
checkpoint_manager = CheckpointManager(
    save_dir= experiment_path / 'save',
    top_k=training_config.checkpoint_k,
    logger=logger)

trainer.callbacks.extend([
    TensorBoardCallback(log_dir= experiment_path / 'tensorboard_logs'),
    CheckpointCallback(manager=checkpoint_manager),
    EarlyStoppingCallback(patience=10),
])

# ── Evaluators ────────────────────────────────────────────────────────────────
trainer.evaluators.extend([
    MSEEvaluator(make_dataloader(data_config)),
    #ODEEvaluator(make_dataloader(data_config)),
])

# ── Run ───────────────────────────────────────────────────────────────────────
## ── DataLoader factory ───────────────────────────────────────────────────────
data_loader = make_dataloader(data_config)
trainer.fit(
    dataloader=data_loader,
    epochs=training_config.epochs,
)