from dataclasses import dataclass
from enum import StrEnum
from pydantic import BaseModel



class ParametersTraining(BaseModel):
    l_r: float = 1e-3
    epochs: int = 2000
    top_k_save_frequency: int = 5 # save every 50 epochs


class AvailablesODE(StrEnum):
    LOTKA_VOLTERA = 'Lotka-Voltera'


class AvailablesAIModel(StrEnum):
    BASIC_PINN = 'BasicPINNS'


class ODESpecifications(BaseModel):
    ode_name: AvailablesODE = AvailablesODE.LOTKA_VOLTERA
    initial_conditions: list[float] = None
    model_dimension: int = 2
    grid_size: int = 200 # Nb de point 
    t_span: tuple[float, float] = (0., 10.)