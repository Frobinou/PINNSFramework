class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device="cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        self.callbacks = []
        self.evaluators = []    
        self.stop_training = False

        self.epoch_step = 0
        self.last_loss = None
        self.state = {}

    def fit(self, dataloader, epochs):

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

            for batch in dataloader:
                self.optimizer.zero_grad()
                loss_dict = self.loss_fn(self.model, batch, t)

                loss_dict["total"].backward()
                self.optimizer.step()

                epoch_loss += loss_dict["total"].item()

                for cb in self.callbacks:
                    cb.on_batch_end(self, loss_dict)

            epoch_loss /= len(dataloader)
            self.last_loss = epoch_loss
            self.state["loss"] = {
                "total": loss_dict["total"].detach(),
                "ode": loss_dict.get("ode"),
                "data": loss_dict.get("data"),
                "residuals": loss_dict.get("residuals"),
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