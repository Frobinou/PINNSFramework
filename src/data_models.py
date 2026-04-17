try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass

from pathlib import Path
from pydantic import BaseModel, model_validator, Field
from typing import Optional, List, Dict, Any

class AvailablesODE(StrEnum):
    LOTKA_VOLTERA = 'Lotka-Voltera'
    CFAST = "CFAST"


class AvailablesAIModel(StrEnum):
    BASIC_PINN = 'BasicPINNS'

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
    initial_conditions: list[float] | None  = None
    model_dimension: int 
    input_cols: list[str] | None  = None
    target_cols: list[str] | None  = None 

    data_config: DataConfig | None = None
    ode_config: ODESConfig | None = None

    @model_validator(mode="after")
    def validate_ic_length(self):
        if self.initial_conditions is not None:
            if len(self.initial_conditions) != self.model_dimension:
                raise ValueError(
                    f"Initial conditions length ({len(self.initial_conditions)}) "
                    f"must match model dimension ({self.model_dimension})"
                )
        return self
    



