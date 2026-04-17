# src/configs.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from src.data_models import ODEExperiment, TrainingConfig, AvailablesODE, AvailablesAIModel, ODESConfig, DataConfig

class TrainingConfig(BaseModel):
    l_r: float = 1e-3
    epochs: int = 2000
    top_k_save_frequency: int = 5 # save every 50 epochs
    log_frequency: int = 50 # log every 50 epochs
    model_name: AvailablesAIModel = AvailablesAIModel.BASIC_PINN
    optimizer: str = "Adam"

class DataConfig(BaseModel):
    data_folder: Path
    lambda_data: float = 1.0    
    batch_size: int = 64
    shuffle: bool = True

class ODESConfig(BaseModel):
    parameters: BaseModel
    ode_name: AvailablesODE = AvailablesODE.LOTKA_VOLTERA
    grid_size: int = 200  # Nb de points
    t_span: tuple[float, float] = (0.0, 10.0)
    lambda_ode: float = 1.0
class ODEExperiment(BaseModel):
    ode_name: str 
    initial_conditions: Optional[list[float]] = None
    model_dimension: int 
    input_cols: Optional[list[str]] = None
    target_cols: Optional[list[str]] = None 

    data_config: Optional[DataConfig] = None
    ode_config: Optional[ODESConfig] = None

class TrainConfig(BaseModel):

    # ODEExperiment parameters
    ode_name: str
    initial_conditions: List[float]
    model_dimension: int
    input_cols: Optional[List[str]] = None
    target_cols: Optional[List[str]] = None 

    model: str = "basic_pinn"

    lr: float = 1e-3
    epochs: int = 2000
    grid_size: int = 200

    
    

    t_span: List[float] = Field(default_factory=lambda: [0.0, 10.0])
    ode_params: Dict[str, float] = Field(default_factory=dict)

    output_dir: str = "runs"
    device: Optional[str] = None

    def validate(self):
        if len(self.initial_conditions) != self.model_dimension:
            raise ValueError("Initial conditions must match model dimension")


class InferConfig(BaseModel):
    experiment_dir: str
    plot: bool = False
    save_plot: bool = False
    device: Optional[str] = None


class GenerateConfig(BaseModel):
    ode: str
    ode_params: Dict[str, float] = Field(default_factory=dict)

    output_file: str = "data/generated_dataset.parquet"

    n_sims: int = 50
    t_max: float = 10.0
    n_steps: int = 200
    seed: int = 42