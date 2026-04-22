from pathlib import Path
from pydantic import BaseModel, Field, model_validator

from src.repositories.odes import AvailablesODE
from src.repositories.models import AvailablesAIModel
from src.repositories.losses import AvailablesLoss

# ── ODE ───────────────────────────────────────────────────────────────────────

class ODESConfig(BaseModel):
    """Physics side: ODE definition + simulation grid."""
    ode_name:   AvailablesODE | None = None
    parameters: BaseModel                       
    t_span:     tuple[float, float]         = (0.0, 10.0)
    grid_size:  int                         = Field(1000, gt=0)
    initial_conditions: list[float]         = Field(default_factory=list)   
    dimension: int                         = Field(1, gt=0)  # for vector-valued ODEs  
     
    @model_validator(mode="after")
    def check_initial_conditions(self) -> "ODESConfig":
        if len(self.initial_conditions) != self.dimension:
            raise ValueError(
                f"initial_conditions has {len(self.initial_conditions)} elements "
                f"but dimension={self.dimension}."
            )
        return self

class PhysicsWeights(BaseModel):
    """Loss weights."""
    name: AvailablesLoss = AvailablesLoss.PINN_LOSS
    lambda_ode:  float = Field(1.0, ge=0.0)
    lambda_data: float = Field(1.0, ge=0.0)


# ── Data ──────────────────────────────────────────────────────────────────────

class DataConfig(BaseModel):
    """Everything needed to build a DataLoader, nothing more."""
    type: str = "parquet"  # for now we only support parquet, but this allows to easily add more data sources in the future
    data_path: Path
    input_cols:   list[str]
    target_cols:  list[str]
    batch_size:   int   = Field(64,  gt=0)
    train_ratio:  float = Field(0.7, gt=0.0, lt=1.0)
    val_ratio:    float = Field(0.15, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def check_ratios(self) -> "DataConfig":
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")
        return self


# ── Training ──────────────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    """Pure training hyperparameters."""
    epochs:        int   = Field(2000, gt=0)
    lr:            float = Field(1e-3, gt=0.0)
    log_frequency: int   = Field(50,   gt=0)
    checkpoint_k:  int   = Field(5,    gt=0)
    model_name:    AvailablesAIModel = AvailablesAIModel.BASIC_PINN
    optimizer:     str = "Adam"  # for now we only support Adam, but this allows to easily add more optimizers in the future

# ── Top-level experiment snapshot ─────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """Single object to dump / reload a full experiment."""
    ode:      ODESConfig
    data:     DataConfig
    model:    AvailablesAIModel
    physics_weights:  PhysicsWeights = PhysicsWeights()
    training: TrainingConfig  = TrainingConfig()