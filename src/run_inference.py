import sys
from pathlib import Path

cwd = str(Path.cwd())
sys.path.append(cwd)
import torch

from src.core.factory import run_inference_from_config
from src.logger import setup_logger
logger = setup_logger()

# ── Force registration ────────────────────────────────────────────────────────
from src.core.bootstrap import bootstrap_registry
bootstrap_registry()
logger.info("Registries bootstrapped successfully.")

# ── Device & output ───────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

experiment_folder_path = Path("runs") / "Lotka-Voltera" / "experiment_2026-04-23_22-03-05"
experiment_path_config = experiment_folder_path / "training_conf.json"
checkpoint_path = experiment_folder_path / "save" / "epoch_97_loss_0.121058.pt"
result = run_inference_from_config(experiment_path=experiment_path_config,
                                   checkpoint_path=checkpoint_path,
                                   device=device)

logger.info(f'results: {result}')