import sys
from pathlib import Path
cwd = str(Path.cwd())
sys.path.append(cwd)

import json
import torch
import matplotlib.pyplot as plt

from pydantic import BaseModel
from src.data_models import ODESpecifications, AvailablesODE, AvailablesAIModel
from src.trainer_runner import ODE_REPOSITORY, AIMODEL_REPOSITORY
from src.ode_repository.ode_lotka_voltera import ParamsLotkaVoltera

class InferenceRunner:

    def __init__(self, experiment_dir, device=None, 
                 ode_specifications: ODESpecifications = None, 
                 ode_parameters: BaseModel = None, 
                 model_name:str = ''
                 ):
        self.experiment_dir = Path(experiment_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ode_specifications: ODESpecifications = ode_specifications
        self.ode_parameters = ode_parameters

        # Model
        self.ode = ODE_REPOSITORY.get(ode_specifications.ode_name)(params=ode_parameters)
        self.model = AIMODEL_REPOSITORY.get(model_name)(output_dim=ode_specifications.model_dimension).to(self.device)


    @classmethod
    def from_config(cls, experiment_dir, config_name="training_config.json"):
        config_path = experiment_dir / config_name

        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} not found")

        with open(config_path, "r") as f:
            config = json.load(f)

        ode_specs = ODESpecifications(**config.get("ode_specifications"))

        if ode_specs.ode_name == AvailablesODE.LOTKA_VOLTERA:
            ode_parameters = ParamsLotkaVoltera(**config.get("ode_parameters"))

        print(config.get('model_name'))
        return cls(experiment_dir=experiment_dir,
                   ode_specifications=ode_specs,
                   ode_parameters = ode_parameters,
                   model_name = AvailablesAIModel(config.get('model_name'))
                   )

    # -------------------------
    # MODEL FACTORY
    # -------------------------
    def build_model(self):
        self.model.to(self.device)
        self.model.eval()

    # -------------------------
    # LOAD WEIGHTS
    # -------------------------
    def load_weights(self):
        save_dir = self.experiment_dir / 'save'
        checkpoint_files = list(save_dir.glob("*.pt"))

        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint files found")

        best_loss = float("inf")
        best_checkpoint = None
        best_path = None

        for path in checkpoint_files:
            checkpoint = torch.load(path, map_location="cpu")

            loss = checkpoint.get("loss")
            if loss is None:
                continue

            if loss < best_loss:
                best_loss = loss
                best_checkpoint = checkpoint
                best_path = path

        if best_checkpoint is None:
            raise ValueError("No valid checkpoint found")

        self.model.load_state_dict(best_checkpoint["model_state_dict"])

        print(f"Loaded best checkpoint: {best_path} (loss={best_loss:.6f})")

    # -------------------------
    # FULL PIPELINE
    # -------------------------
    def load(self):
        self.build_model()
        self.load_weights()

    # -------------------------
    # PREDICTION
    # -------------------------
    def predict(self):
        t0, t1 = self.ode_specifications.t_span

        t = torch.linspace(t0, t1, self.ode_specifications.grid_size).view(-1, 1)
        t = t.to(self.device)

        with torch.no_grad():
            pred = self.model(t)

        return t.cpu().numpy(), pred.cpu().numpy()

    # -------------------------
    # PLOT
    # -------------------------
    def plot(self, save=True):
        t, pred = self.predict()

        plt.figure(figsize=(8, 5))

        for i in range(pred.shape[1]):
            plt.plot(t, pred[:, i], label=f"State {i}")

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Model Inference")
        plt.legend()
        plt.grid()

        if save:
            save_path = self.experiment_dir / "inference.png"
            plt.savefig(save_path)
            print(f"Saved plot: {save_path}")

        plt.show()

if __name__ == "__main__":
    model_path = Path("runs\Lotka-Voltera\experiment_2026-04-16_15-13-10")
    runner = InferenceRunner.from_config(experiment_dir=model_path)
    runner.load()
    runner.plot()