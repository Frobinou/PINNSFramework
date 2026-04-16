import json
import torch
from pathlib import Path
from typing import List, Tuple


class CheckpointManager:
    def __init__(self, save_dir: Path, top_k: int, logger):
        self.save_dir = save_dir
        self.top_k = top_k
        self.logger = logger
        self.best_checkpoints: List[Tuple[float, Path]] = []

    def save_top_k_checkpoint(self, epoch: int, loss: float, model, optimizer, global_step: int):
        checkpoint_path = self.save_dir / f"epoch_{epoch}_loss_{loss:.6f}.pt"

        # Sauvegarde
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "global_step": global_step,
        }, checkpoint_path)

        # Ajout à la liste
        self.best_checkpoints.append((loss, checkpoint_path))

        # Tri par loss croissante
        self.best_checkpoints.sort(key=lambda x: x[0])

        # Si on dépasse top_k → supprimer le pire
        if len(self.best_checkpoints) > self.top_k:
            worst_loss, worst_path = self.best_checkpoints.pop(-1)

            if worst_path.exists():
                worst_path.unlink()
                self.logger.info(f"Removed worst checkpoint: {worst_path}")

        self.logger.info(f"Saved checkpoint (top-{self.top_k}): {checkpoint_path}")

    def load_checkpoint(self, path: Path, model, optimizer):
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint.get("global_step", 0)

        self.logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")

        return global_step

    def save_config(self, experiment_folder: Path, model_name: str, parameters_training, ode_specifications, ode_parameters, optimizer):
        config = {
            "model_name": model_name,
            "parameters_training": parameters_training.model_dump(mode="json"),
            "ode_specifications": ode_specifications.model_dump(mode="json"),
            "ode_parameters": ode_parameters.model_dump(),
            "optimizer": optimizer.__class__.__name__,
        }

        config_path = experiment_folder / "training_config.json"

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        self.logger.info(f"Config saved: {config_path}")