import torch
from torch.utils.tensorboard import SummaryWriter
from pydantic import BaseModel
from datetime import datetime
import json

from src.data_models import ParametersTraining, ODESpecifications, AvailablesODE, AvailablesAIModel
from src.logger import setup_logger
from model_repository.model_PINN import BasicPINN
from ode_repository.ode_lotka_voltera import LotkaVoltera


ODE_REPOSITORY = {
    AvailablesODE.LOTKA_VOLTERA : LotkaVoltera
}

AIMODEL_REPOSITORY = {
    AvailablesAIModel.BASIC_PINN : BasicPINN
}

class Trainer:
    def __init__(
        self,
        parameters_training: ParametersTraining,
        ode_specifications:ODESpecifications,
        ode_parameters:BaseModel,
        device:str = "",
        output_folder_path:str = "",
        model_name:str = ""
        
    ):
        self.logger = setup_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f'Device found: {self.device}')
        self.parameters_training: ParametersTraining = parameters_training
        self.ode_specifications: ODESpecifications = ode_specifications
        self.ode_parameters = ode_parameters

        # Model
        self.ode = ODE_REPOSITORY.get(ode_specifications.ode_name)(params=ode_parameters)
        self.model = AIMODEL_REPOSITORY.get(model_name)(output_dim=ode_specifications.model_dimension).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.parameters_training.l_r)

        # Logger
        ## Create a new director
        experiment_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_folder = output_folder_path / f"experiment_{experiment_date}"
        self.experiment_folder.mkdir(parents=True, exist_ok=True)

        self.save_config()

        log_dir = self.experiment_folder / 'tensorboard_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # About saving models
        self.save_dir = self.experiment_folder / 'save'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k: int = parameters_training.top_k_save_frequency
        self.best_checkpoints = []  # liste de tuples (loss, path)

        self.global_step = 0

        
        # Time grid
        self.t = torch.linspace(
            0,
            self.ode_specifications.model_dimension,
            self.ode_specifications.grid_size,
            device=self.device
            ).view(-1, 1)

        self.t.requires_grad = True
        # Boundary conditions
        self.x0 = torch.tensor(
            self.ode_specifications.initial_conditions,
            dtype=torch.float32,
            device=self.device
        )

    # ---------- Core computations ----------
    def compute_derivative(self, y_pred):
        return torch.cat([
            torch.autograd.grad(
                y_pred[:, i],
                self.t,
                torch.ones_like(y_pred[:, i]),
                create_graph=True,
                retain_graph=True
            )[0]
            for i in range(self.ode_specifications.model_dimension)
        ], dim=1)

    def compute_loss(self):
        alpha = 0.9
        y_pred = self.model(self.t)

        gradient_derivative = self.compute_derivative(y_pred)
        F_t = self.ode.torch_ode(y_pred)

        #ode_loss = torch.mean((gradient_derivative - F_t) ** 2) # f'(t) = F(f,t)
        ode_loss_prey = torch.mean((gradient_derivative[:, 0] - F_t[:, 0]) ** 2)
        ode_loss_pred = torch.mean((gradient_derivative[:, 1] - F_t[:, 1]) ** 2)
        ode_loss = alpha * ode_loss_prey + (1-alpha)* ode_loss_pred 
        # initial condition at t=0
        pred0 = self.model(torch.zeros(1, 1, device=self.device))
        ic_loss = torch.mean((pred0 - self.x0) ** 2)

        return ode_loss + ic_loss, ode_loss, ic_loss, ode_loss_prey, ode_loss_pred

    # ---------- Training ----------
    def train(self):
        for epoch in range(self.parameters_training.epochs):
            
            self.global_step += 1
            self.optimizer.zero_grad()

            loss, ode_loss, ic_loss, ode_loss_prey, ode_loss_pred = self.compute_loss()

            loss.backward()
            self.optimizer.step()

            if epoch % 50 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                self.writer.add_scalar("Loss/train/total_loss", loss.item(), epoch)
                self.writer.add_scalar("Loss/train/ode_loss", ode_loss.item(), epoch)
                self.writer.add_scalar("Loss/train/ic_loss", ic_loss.item(), epoch)
                self.writer.add_scalar("Loss/train/ode_loss_prey", ode_loss_prey.item(), epoch)
                self.writer.add_scalar("Loss/train/ode_loss_pred", ode_loss_pred.item(), epoch)
                
                self.evaluate(epoch=epoch)
                
                if len(self.best_checkpoints) < self.top_k or loss.item() < self.best_checkpoints[-1][0]:
                    self.save_top_k_checkpoint(epoch, loss.item())

    # ---------- Evaluation ----------
    def evaluate(self, epoch:int):
        self.model.eval()

        sol = self.ode.validation(
            t_span=self.ode_specifications.t_span,
            x0=self.x0
        )

        t_true = sol.t
        y_true = sol.y.T # Because torch is (N, dim) and scipy (dim, N)

        t_tensor = torch.tensor(t_true, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred = self.model(t_tensor)
            mse = torch.mean((y_pred - y_true_tensor) ** 2)
            self.writer.add_scalar("Eval/MSE", mse.item(), epoch)

        self.log_trajectory_phase_space_plot(y_true=y_true, 
                                            y_pred=y_pred, 
                                            tensorboard_path="Evaluation/Observables/DynamicPhaseTrajectories", 
                                            global_step=epoch )
        self.log_trajectory_plot(t_true, y_true=y_true,
                                 y_pred=y_pred, 
                                 tensorboard_path="Evaluation/Observables/DynamicTrajectories", 
                                 global_step=epoch)
        
    # ---------- Plot ----------
    def log_trajectory_plot(self, t_true,y_true, y_pred, global_step:int, tensorboard_path):
        img = self.ode.log_trajectory_plot(t_true=t_true, y_true=y_true, y_pred=y_pred)
        self.writer.add_image(tensorboard_path, img, global_step)


    # ---------- Simple visualization ----------
    def log_trajectory_phase_space_plot(self,y_true, y_pred, global_step:int, tensorboard_path: str):
        img = self.ode.log_trajectory_phase_space_plot(y_true=y_true, y_pred=y_pred)
        self.writer.add_image(tensorboard_path, img, global_step)
        

    # ---------- Run all ----------
    def run(self):
        self.train()

        # Final evaluation
        self.evaluate(epoch=self.global_step)

        self.writer.close()

    def save_top_k_checkpoint(self, epoch, loss):
        checkpoint_path = self.save_dir / f"epoch_{epoch}_loss_{loss:.6f}.pt"

        # Sauvegarde
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "global_step": self.global_step,
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


    def load_checkpoint(self, path):

        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)

        self.logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")

    def save_config(self):
        config = {
            "model_name": self.model.name,
            "parameters_training": self.parameters_training.model_dump(mode="json"),
            "ode_specifications": self.ode_specifications.model_dump(mode="json"),
            "ode_parameters": self.ode_parameters.model_dump(),
            "optimizer": self.optimizer.__class__.__name__,
        }

        config_path = self.experiment_folder / "training_config.json"

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        self.logger.info(f"Config saved: {config_path}")