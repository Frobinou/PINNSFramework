import torch
from torch.utils.tensorboard import SummaryWriter
from pydantic import BaseModel
from datetime import datetime
import json

from src.data_models import ParametersTraining, ODESpecifications, AvailablesODE, AvailablesAIModel
from src.logger import setup_logger
from src.core.checkpoint_manager import CheckpointManager

from src.model_repository.model_PINN import BasicPINN
from src.ode_repository.ode_lotka_voltera import LotkaVoltera
from src.ode_repository.ode_cfast import ODECFAST


ODE_REPOSITORY = {
    AvailablesODE.LOTKA_VOLTERA : LotkaVoltera,
    AvailablesODE.CFAST : ODECFAST
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

        

        log_dir = self.experiment_folder / 'tensorboard_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # About saving models
        self.save_dir = self.experiment_folder / 'save'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k: int = parameters_training.top_k_save_frequency
        self.checkpoint_manager = CheckpointManager(self.save_dir, self.top_k, self.logger)
        self.global_step = 0
        self.log_frequency = parameters_training.log_frequency

        
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

    def loss_fn(self):
        alpha = 0.9
        y_pred = self.model(self.t)

        gradient_derivative = self.compute_derivative(y_pred)
        F_t = self.ode.torch_ode(y_pred)

        #ode_loss = torch.mean((gradient_derivative - F_t) ** 2) # f'(t) = F(f,t)
        all_ode_loss_dim = []
        ode_loss = 0.
        for i in range(self.ode_specifications.model_dimension):
            ode_loss_dim = torch.mean((gradient_derivative[:, i] - F_t[:, i]) ** 2)
            ode_loss += ode_loss_dim
            all_ode_loss_dim.append(ode_loss_dim)
        # initial condition at t=0
        pred0 = self.model(torch.zeros(1, 1, device=self.device))
        ic_loss = torch.mean((pred0 - self.x0) ** 2)

        return ode_loss + ic_loss, ode_loss, ic_loss, all_ode_loss_dim

    def train(self):
        for epoch in range(self.parameters_training.epochs):
            self.global_step += 1
            self.optimizer.zero_grad()

            loss, ode_loss, ic_loss, all_ode_loss_dim = self.loss_fn()

            loss.backward()
            self.optimizer.step()

            if epoch % self.log_frequency == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                self.writer.add_scalar("Loss/train/total_loss", loss.item(), epoch)
                self.writer.add_scalar("Loss/train/ode_loss", ode_loss.item(), epoch)
                self.writer.add_scalar("Loss/train/ic_loss", ic_loss.item(), epoch)
                for i, loss_dim in enumerate(all_ode_loss_dim):
                    self.writer.add_scalar(f"Loss/train/ode_loss_{i}", loss_dim.item(), epoch)

                self.evaluate(epoch)

                if len(self.checkpoint_manager.best_checkpoints) < self.parameters_training.top_k_save_frequency or loss.item() < self.checkpoint_manager.best_checkpoints[-1][0]:
                    self.checkpoint_manager.save_top_k_checkpoint(epoch, loss.item(), self.model, self.optimizer, self.global_step)

        return self.global_step

    # ---------- Evaluation ----------

    
    def evaluate(self, epoch:int):
        self.model.eval()

        sol = self.ode.validation(
            t_span=self.ode_specifications.t_span,
            x0=self.x0, 
            nb_points=self.ode_specifications.grid_size
        )

        if len(sol.t) == 0 or len(sol.y) == 0:
            self.logger.info(f'The validation solver did not return any solution at epoch {epoch}. Skipping evaluation. Check the ODE specifications and parameters.')
            return
        
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
        if img is not None:
            self.writer.add_image(tensorboard_path, img, global_step)


    # ---------- Simple visualization ----------
    def log_trajectory_phase_space_plot(self,y_true, y_pred, global_step:int, tensorboard_path: str):
        img = self.ode.log_trajectory_phase_space_plot(y_true=y_true, y_pred=y_pred)

        if img is not None:
            self.writer.add_image(tensorboard_path, img, global_step)
        

    # ---------- Run all ----------
    def run(self):
        self.save_config()

        self.train()

        # Final evaluation
        self.evaluate(epoch=self.global_step)

        self.writer.close()

    def save_top_k_checkpoint(self, epoch, loss):
        self.checkpoint_manager.save_top_k_checkpoint(epoch, loss, self.model, self.optimizer, self.global_step)


    def load_checkpoint(self, path):
        self.global_step = self.checkpoint_manager.load_checkpoint(path, self.model, self.optimizer)

    def save_config(self):
        self.checkpoint_manager.save_config(self.experiment_folder, self.model.name, self.parameters_training, self.ode_specifications, self.ode_parameters, self.optimizer)