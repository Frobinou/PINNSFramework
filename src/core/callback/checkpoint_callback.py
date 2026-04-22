from src.core.callback.base_callback import Callback
from src.core.checkpoint_manager import CheckpointManager


class CheckpointCallback(Callback):
    """Saves top-k checkpoints at the end of each epoch."""

    def __init__(self, manager: CheckpointManager):
        self.manager = manager

    def on_epoch_end(self, trainer, epoch: int) -> None:
        self.manager.save_top_k_checkpoint(
            epoch=epoch,
            loss=trainer.last_loss,
            model=trainer.model,
            optimizer=trainer.optimizer,
            global_step=trainer.epoch,
        )

    def on_train_end(self, trainer):
        self.writer.close()

    def on_batch_end(self, trainer, loss):
        return super().on_batch_end(trainer, loss)
    
    def on_evaluation_end(self, trainer, evaluation_results):
        return super().on_evaluation_end(trainer, evaluation_results)

    def on_epoch_start(self, trainer, epoch):
        return super().on_epoch_start(trainer, epoch)
    
    def on_train_start(self, trainer):
        return super().on_train_start(trainer)