import torch
from src.repositories.data_loader.base_dataloader import BaseDataLoader

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        t: torch.Tensor,
        device="cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.t = t

        self.callbacks = []
        self.evaluators = []    
        self.stop_training = False

        self.epoch_step = 0
        self.last_loss = None
        self.state = {}

        self.dataloader = None

    def fit(self, dataloader: BaseDataLoader, epochs: int) -> None:
        self.dataloader = dataloader

        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(epochs):
            self._reset_epoch_state()
            if self.stop_training:
                break

            self.epoch_step += 1
            self.model.train()

            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            epoch_loss = 0.0
            num_batches = 0
            loss_dict = {}

            for batch in self.dataloader.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                loss_dict = self.loss_fn(self.model, batch,self.t.to(self.device))
                loss_dict["total"].backward()
                self.optimizer.step()

                epoch_loss += loss_dict["total"].item()
                num_batches += 1

                for cb in self.callbacks:
                    cb.on_batch_end(self, loss_dict)

            if num_batches > 0:
                epoch_loss /= num_batches

            self.last_loss = epoch_loss
            self.state["loss"] = {
                "total": loss_dict.get("total", torch.tensor(0.0)).detach(),
                "ode": loss_dict.get("ode").detach() if loss_dict.get("ode") is not None else None,
                "data": loss_dict.get("data").detach() if loss_dict.get("data") is not None else None,
                "residuals": loss_dict.get("residuals").detach() if loss_dict.get("residuals") is not None else None,
            }

            self.state["epoch"] = self.epoch_step

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch)

            for ev in self.evaluators:
                ev.run(self)

        for cb in self.callbacks:
            cb.on_train_end(self)

    def _reset_epoch_state(self):
        self.state["loss"] = {}
        self.state["metrics"] = {}