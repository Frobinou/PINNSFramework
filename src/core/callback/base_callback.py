from abc import ABC, abstractmethod

class Callback(ABC):
    @abstractmethod
    def on_train_start(self, trainer): pass

    @abstractmethod
    def on_epoch_start(self, trainer, epoch): pass

    @abstractmethod
    def on_batch_end(self, trainer, loss): pass

    @abstractmethod
    def on_epoch_end(self, trainer, epoch): pass
    
    @abstractmethod 
    def on_train_end(self, trainer): pass