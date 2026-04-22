# ── checkpoint_manager.py ─────────────────────────────────────────────────────

import json
import logging
import torch
from pathlib import Path

from src.core.schemas import ExperimentConfig


class CheckpointManager:
    """Handles top-k checkpoint saving, loading, and config serialisation."""

    def __init__(self, save_dir: Path, top_k: int, logger: logging.Logger):
        self.save_dir = save_dir
        self.top_k = top_k
        self.logger = logger
        self.best_checkpoints: list[tuple[float, Path]] = []

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_top_k_checkpoint(
        self,
        epoch: int,
        loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        global_step: int,
    ) -> None:
        path = self.save_dir / f"epoch_{epoch}_loss_{loss:.6f}.pt"

        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                 loss,
                "global_step":          global_step,
            },
            path,
        )

        self.best_checkpoints.append((loss, path))
        self.best_checkpoints.sort(key=lambda x: x[0])

        if len(self.best_checkpoints) > self.top_k:
            _, worst_path = self.best_checkpoints.pop(-1)
            if worst_path.exists():
                worst_path.unlink()
                self.logger.info(f"Removed worst checkpoint: {worst_path}")

        self.logger.info(f"Saved checkpoint (top-{self.top_k}): {path}")

    def load_checkpoint(
        self,
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> int:
        """Restore model & optimizer weights. Returns global_step."""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        self.logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return global_step

    def save_config(self, experiment: ExperimentConfig) -> None:
        """Serialise the full experiment config to save_dir/config.json."""
        path = self.save_dir / "config.json"
        path.write_text(experiment.model_dump_json(indent=4))
        self.logger.info(f"Config saved: {path}")