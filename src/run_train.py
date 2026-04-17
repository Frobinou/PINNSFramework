import sys
from pathlib import Path
cwd = str(Path.cwd())
sys.path.append(cwd)


from pathlib import Path
import torch

from src.core.trainer_runner import Trainer
from src.data_models import DataConfig, TrainingConfig, ODEExperiment, AvailablesODE, AvailablesAIModel, ODESConfig


device = "cuda" if torch.cuda.is_available() else "cpu"
output_folder_path = Path("runs") / "Lotka-Voltera"

# Define training configuration
training_config = TrainingConfig(l_r=1e-3, 
                                 epochs=2000, 
                                 top_k_save_frequency=5, 
                                 log_frequency=50,
                                model_name=AvailablesAIModel.BASIC_PINN, 
                                optimizer="Adam")

# ODE Parameters
from src.odes.ode_repository.ode_lotka_voltera import ParamsLotkaVoltera
Pa = ParamsLotkaVoltera(alpha=1.5, beta=1.0, delta=0.75, gamma=1.0)
ode_config = ODESConfig(      
        parameters=Pa,  
        ode_name=AvailablesODE.LOTKA_VOLTERA,
        grid_size=200,
        t_span=(0.0, 10.0),
        lambda_ode=0.0
    )

# Data Configuration
data_config = DataConfig(
        data_folder=Path("data/lotka_volterra.parquet"),
        lambda_data=1.0,
        batch_size=64,
        shuffle=True
    )

# Experiment Configuration
ode_experiment_config = ODEExperiment(
    ode_name="Lotka-Voltera",   
    initial_conditions=[10.0, 10.0],
    model_dimension=2,  
    input_cols=["t"], # "alpha", "beta", "delta", "gamma" not yet used as inputs, but could be in a more complex scenario            
    target_cols=['prey', 'predator'],   
    ode_config=ode_config,  
    data_config=data_config
)

trainer = Trainer(
    training_config=training_config,
    ode_experiment_config=ode_experiment_config,
    device=device,
    output_folder_path=output_folder_path,                                   
    )
trainer.run()