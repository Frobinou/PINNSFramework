from src.core.callback.base_callback import Callback

class LoggingCallback(Callback):
    def __init__(self, logger, freq=10):
        self.logger = logger
        self.freq = freq

    def on_epoch_end(self, trainer, epoch):
        if trainer.epoch % self.freq == 0:
            self.logger.add_scalar(
                "Training/loss",
                trainer.last_loss,
                trainer.epoch
            )