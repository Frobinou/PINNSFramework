import torch
from torch.utils.tensorboard import SummaryWriter
from pydantic import BaseModel
from datetime import datetime
import json
from pathlib import Path

from src.data_models import TrainingConfig, ODESConfig, DataConfig, ODEExperiment
from src.logger import setup_logger
from src.core.checkpoint_manager import CheckpointManager
from src.core.data_loader import ParquetDataLoader
from src.ai_model_repository import AIMODEL_REPOSITORY
from src.odes import ODE_REPOSITORY


class Trainer:
    def __init__(
        self,
        training_config: TrainingConfig,
        ode_experiment_config: ODEExperiment,
        device:str = "",
        output_folder_path:Path = Path()
    ):
        self.logger = setup_logger()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f'Device found: {self.device}')

        self.ode_experiment_config: ODEExperiment = ode_experiment_config
        self.training_config: TrainingConfig = training_config

        self._initialize_training_workspace(output_folder_path)
        self._initialize_training_environment()
        self._initalize_dataloader()
        self._initalize_ode_environment()
        
    def _initalize_dataloader(self):
        data_config:DataConfig = self.ode_experiment_config.data_config
        if data_config is not None:
            self.logger.info(f"Data folder found at {self.ode_experiment_config.data_config.data_folder}. Initializing dataloader.")
            self.lambda_data = data_config.lambda_data
            loader_builder = ParquetDataLoader(
                parquet_path=data_config.data_folder,
                input_cols=self.ode_experiment_config.input_cols,
                target_cols=self.ode_experiment_config.target_cols,
                batch_size=data_config.batch_size,
                shuffle=data_config.shuffle
            )
            self.dataloader = loader_builder.get_loader()
            self.data_iter = iter(self.dataloader)
        else:
            self.lambda_data = 0.0  

    def _initialize_training_workspace(self, output_folder_path:Path):
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

        self.top_k: int = self.training_config.top_k_save_frequency
        self.log_frequency = self.training_config.log_frequency

        self.checkpoint_manager = CheckpointManager(self.save_dir, self.top_k, self.logger)
        self.global_step = 0
        
    def _initalize_ode_environment(self):
        ode_config = self.ode_experiment_config.ode_config
        if ode_config is not None:
            self.ode = ODE_REPOSITORY.get(ode_config.ode_name)(params=ode_config.parameters)  
            self.lambda_ode = ode_config.lambda_ode
        else:
            self.lambda_ode = 0.0

    def _initialize_training_environment(self):

        # Model
        self.model = AIMODEL_REPOSITORY.get(self.training_config.model_name)(output_dim=self.ode_experiment_config.model_dimension).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config.l_r)

        # Time grid
        ode_config = self.ode_experiment_config.ode_config

        self.t = torch.linspace(
            0,
            self.ode_experiment_config.model_dimension,
            ode_config.grid_size,
            device=self.device
            ).view(-1, 1)

        self.t.requires_grad = True
        # Boundary conditions
        self.x0 = torch.tensor(
            self.ode_experiment_config.initial_conditions,
            dtype=torch.float32,
            device=self.device
        )

    def _get_data_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        return batch

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
            for i in range(self.ode_experiment_config.model_dimension)
        ], dim=1)

    def physics_loss(self):
        y_pred = self.model(self.t)

        gradient_derivative = self.compute_derivative(y_pred)
        F_t = self.ode.torch_ode(y_pred)

        #ode_loss = torch.mean((gradient_derivative - F_t) ** 2) # f'(t) = F(f,t)
        all_ode_loss_dim = []
        ode_loss = 0.
        for i in range(self.ode_experiment_config.model_dimension):
            ode_loss_dim = torch.mean((gradient_derivative[:, i] - F_t[:, i]) ** 2)
            ode_loss += ode_loss_dim
            all_ode_loss_dim.append(ode_loss_dim)
        # initial condition at t=0
        pred0 = self.model(torch.zeros(1, 1, device=self.device))
        ic_loss = torch.mean((pred0 - self.x0) ** 2)

        return ode_loss + ic_loss, ode_loss, ic_loss, all_ode_loss_dim

    def compute_data_loss(self, batch):
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)

        y_pred = self.model(x)

        return torch.mean((y_pred - y) ** 2)
    
    def train(self):
        for epoch in range(self.training_config.epochs):
            self.global_step += 1
            self.optimizer.zero_grad()

            # -------------------------
            # Physics loss
            # -------------------------
            loss = 0.0
            if self.lambda_ode > 0:
                loss_ode, ode_loss, ic_loss, all_ode_loss_dim = self.physics_loss()
                loss += self.lambda_ode * loss_ode

            # -------------------------
            # Data loss 
            # -------------------------
            if self.lambda_data > 0:
                batch = self._get_data_batch()
                loss_data = self.compute_data_loss(batch)
                loss += self.lambda_data * loss_data


            # Backprop
            loss.backward()
            self.optimizer.step()

            if epoch % self.log_frequency == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                self.logger.info(f'here: {self.lambda_ode}')
                if self.lambda_ode > 0:
                    self.writer.add_scalar(f"Loss_physics/total", ode_loss.item(), epoch)
                    for i, loss_dim in enumerate(all_ode_loss_dim):
                        self.writer.add_scalar(f"Loss_physics/ode_loss_{self.ode_experiment_config.target_cols[i]}", loss_dim.item(), epoch)
                        #self.logger.info(f"    ODE Loss dim {i}: {loss_dim.item():.6f}")

                if self.lambda_data > 0:
                    self.writer.add_scalar("Loss_data/MSE", loss_data.item(), epoch)


                self.evaluate(epoch)

                if len(self.checkpoint_manager.best_checkpoints) < self.training_config.top_k_save_frequency or loss.item() < self.checkpoint_manager.best_checkpoints[-1][0]:
                    self.checkpoint_manager.save_top_k_checkpoint(epoch, loss.item(), self.model, self.optimizer, self.global_step)

        return self.global_step

    # ---------- Evaluation ----------

    
    def evaluate(self, epoch:int):
        self.model.eval()

        sol = self.ode.simulate(
            t_span=self.ode_experiment_config.ode_config.t_span,
            x0=self.x0.cpu().numpy(), 
            nb_points=self.ode_experiment_config.ode_config.grid_size
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
                                            y_pred=y_pred.cpu().numpy(), 
                                            tensorboard_path="Evaluation/Observables/DynamicPhaseTrajectories", 
                                            global_step=epoch )
        self.log_trajectory_plot(t_true, y_true=y_true,
                                 y_pred=y_pred.cpu().numpy(), 
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
        self.checkpoint_manager.save_config(experiment_folder=self.experiment_folder, 
                                            ode_experiment_config=self.ode_experiment_config, 
                                            training_config=self.training_config)
        
