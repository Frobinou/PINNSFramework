from enum import StrEnum
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional

class ParametersTraining(BaseModel):
    l_r: float = 1e-3
    epochs: int = 2000
    top_k_save_frequency: int = 5 # save every 50 epochs
    log_frequency: int = 50 # log every 50 epochs


class AvailablesODE(StrEnum):
    LOTKA_VOLTERA = 'Lotka-Voltera'
    CFAST = "CFAST"


class AvailablesAIModel(StrEnum):
    BASIC_PINN = 'BasicPINNS'


class ODESpecifications(BaseModel):
    ode_name: AvailablesODE = AvailablesODE.LOTKA_VOLTERA
    initial_conditions: Optional[list[float]] = None
    model_dimension: int = 2
    grid_size: int = 200  # Nb de points
    t_span: tuple[float, float] = (0.0, 10.0)

    @model_validator(mode="after")
    def validate_ic_length(self):
        if self.initial_conditions is not None:
            if len(self.initial_conditions) != self.model_dimension:
                raise ValueError(
                    f"Initial conditions length ({len(self.initial_conditions)}) "
                    f"must match model dimension ({self.model_dimension})"
                )
        return self