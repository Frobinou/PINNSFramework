from src.core.callback.base_callback import Callback

class EarlyStoppingCallback(Callback):
    def __init__(self, patience=20):
        self.best = float("inf")
        self.counter = 0
        self.patience = patience

    def on_epoch_end(self, trainer, epoch):

        if trainer.last_loss < self.best:
            self.best = trainer.last_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            trainer.stop_training = True

    def on_train_end(self, trainer):
        pass

    def on_train_start(self, trainer):
        pass

    def on_batch_end(self, trainer, loss):
        return super().on_batch_end(trainer, loss)

    def on_epoch_start(self, trainer, epoch):
        return super().on_epoch_start(trainer, epoch)  

