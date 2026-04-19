# PINN Trainer

Physics-Informed Neural Network trainer combining ODE residual loss with empirical data loss.
Built on PyTorch with TensorBoard logging and automatic top-k checkpointing.

---

## Overview

The trainer minimises a composite loss:

$$
\mathcal{L} = \lambda_{\text{ode}} \cdot (\mathcal{L}_{\text{ode}} + \mathcal{L}_{\text{ic}}) + \lambda_{\text{data}} \cdot \mathcal{L}_{\text{data}}
$$

| Term | Description |
|---|---|
| $\mathcal{L}_{\text{ode}}$ | Mean squared ODE residual on the collocation grid |
| $\mathcal{L}_{\text{ic}}$ | Mean squared error on initial conditions at $t=0$ |
| $\mathcal{L}_{\text{data}}$ | Mean squared error on empirical observations (Parquet) |

Three training modes are supported depending on which $\lambda$ values are non-zero:

| Mode | `lambda_ode` | `lambda_data` |
|---|---|---|
| Physics-only | > 0 | 0 |
| Data-only | 0 | > 0 |
| Hybrid PINN | > 0 | > 0 |

---

## Installation

```bash
pip install -r requirements.txt
```

**Required dependencies:**

- `torch >= 2.0`
- `numpy`
- `matplotlib`
- `scipy`
- `tensorboard`
- `pyarrow` / `pandas` (for Parquet data loading)

---

## Project structure

```
src/
├── core/
│   ├── checkpoint_manager.py   # Top-k checkpoint logic
│   ├── data_loader.py          # Parquet DataLoader builder
│   └── models.py               # TrainingStepLog dataclass
├── data_models.py              # TrainingConfig, DataConfig, ODEExperiment
├── ai_model_repository.py      # Neural network registry
├── odes/                       # ODE registry and implementations
├── logger.py                   # Logger setup
└── trainer.py                  # Trainer (this module)
```

---

## Quick start

```python
from pathlib import Path
from src.trainer import Trainer
from src.data_models import TrainingConfig, ODEExperiment

training_config = TrainingConfig(
    epochs=5000,
    l_r=1e-3,
    log_frequency=100,
    top_k_save_frequency=5,
    model_name="MLP",
)

ode_experiment_config = ODEExperiment(
    ode_config=...,         # ODE name, parameters, grid_size, t_span, lambda_ode
    data_config=None,       # set to DataConfig(...) for hybrid mode
    model_dimension=2,
    initial_conditions=[1.0, 0.5],
    input_cols=["t"],
    target_cols=["prey", "predator"],
)

trainer = Trainer(
    training_config=training_config,
    ode_experiment_config=ode_experiment_config,
    output_folder_path=Path("outputs/"),
)

trainer.run()
```

---

## Configuration

### `TrainingConfig`

| Field | Type | Description |
|---|---|---|
| `epochs` | `int` | Number of training epochs |
| `l_r` | `float` | Adam learning rate |
| `log_frequency` | `int` | Epoch interval between TensorBoard writes |
| `top_k_save_frequency` | `int` | Maximum number of checkpoints to keep |
| `model_name` | `str` | Key in `AIMODEL_REPOSITORY` |

### `ODEExperiment`

| Field | Type | Description |
|---|---|---|
| `ode_config` | `ODEConfig \| None` | ODE specification; `None` disables physics loss |
| `data_config` | `DataConfig \| None` | Data specification; `None` disables data loss |
| `model_dimension` | `int` | Number of ODE state variables |
| `initial_conditions` | `list[float]` | Initial state vector $x_0$ |
| `input_cols` | `list[str]` | Input column names in the Parquet file |
| `target_cols` | `list[str]` | Target column names — also used as variable labels in TensorBoard |

### `ODEConfig`

| Field | Type | Description |
|---|---|---|
| `ode_name` | `str` | Key in `ODE_REPOSITORY` |
| `parameters` | `dict` | ODE-specific parameters (e.g. `{"alpha": 1.0, "beta": 0.1}`) |
| `grid_size` | `int` | Number of collocation points |
| `t_span` | `tuple[float, float]` | Integration interval $[t_0, t_f]$ |
| `lambda_ode` | `float` | Weight of the physics loss |

---

## Output directory structure

Each call to `run()` creates a timestamped experiment folder:

```
outputs/
└── experiment_YYYY-MM-DD_HH-MM-SS/
    ├── tensorboard_logs/       # SummaryWriter output
    ├── save/                   # Top-k checkpoints (.pt)
    ├── training_config.json
    └── ode_experiment_config.json
```

Launch TensorBoard with:

```bash
tensorboard --logdir outputs/
```

---

## TensorBoard metrics

### Training scalars

| Tag | Description |
|---|---|
| `Training/loss/total` | Combined weighted loss |
| `Training/loss/physics` | ODE residual loss (when `lambda_ode > 0`) |
| `Training/loss/data` | Data loss (when `lambda_data > 0`) |
| `Training/residuals/<var_name>` | Mean absolute residual per state variable |
| `Training/residuals/mean` | Global mean residual across all points and variables |
| `Training/residuals/max` | Maximum absolute residual — flags poorly satisfied collocation points |
| `Training/gradients/global_norm` | Global L2 gradient norm — diagnoses vanishing/exploding gradients |

### Evaluation scalars

| Tag | Description |
|---|---|
| `Evaluation/MSE` | Global MSE against the scipy reference solution |
| `Evaluation/MSE/<var_name>` | Per-variable MSE (e.g. `Evaluation/MSE/prey`) |

### Evaluation images

| Tag | Description |
|---|---|
| `Evaluation/Observables/DynamicTrajectories` | Time-domain trajectories: predicted vs reference |
| `Evaluation/Observables/DynamicPhaseTrajectories` | Phase-space portrait at current epoch |
| `Evaluation/Observables/PhasePortraitOverlay` | Phase portrait overlay across epochs — visualises convergence toward the attractor |
| `Evaluation/Residuals/CollocationHeatmap` | Heatmap of `\|residual\|` on the collocation grid; one row per variable, time on x-axis |

---

## Example: Lotka-Volterra

```python
ode_config = ODEConfig(
    ode_name="LotkaVolterra",
    parameters={"alpha": 1.0, "beta": 0.1, "delta": 0.075, "gamma": 1.5},
    grid_size=1000,
    t_span=(0.0, 15.0),
    lambda_ode=1.0,
)

ode_experiment_config = ODEExperiment(
    ode_config=ode_config,
    data_config=None,
    model_dimension=2,
    initial_conditions=[10.0, 5.0],
    input_cols=["t"],
    target_cols=["prey", "predator"],  # used as var_names in all TensorBoard tags
)
```

Expected TensorBoard tags for this setup:

- `Training/residuals/prey`, `Training/residuals/predator`
- `Evaluation/MSE/prey`, `Evaluation/MSE/predator`
- `Evaluation/Residuals/CollocationHeatmap` — two rows (prey / predator)
- `Evaluation/Observables/PhasePortraitOverlay` — prey vs predator phase plane

---

## Adding a new ODE

1. Implement a class inheriting from the base ODE interface:

```python
class MyODE:
    def __init__(self, params: dict): ...

    def torch_ode(self, y: torch.Tensor) -> torch.Tensor:
        """Right-hand side F(y, t) — shape (N, n_vars)."""
        ...

    def simulate(self, t_span, x0, nb_points):
        """scipy reference solution."""
        ...

    def log_trajectory_plot(self, t_true, y_true, y_pred) -> np.ndarray | None:
        """Return a CHW uint8 array or None to skip logging."""
        ...

    def log_trajectory_phase_space_plot(self, y_true, y_pred) -> np.ndarray | None:
        ...
```

2. Register it:

```python
ODE_REPOSITORY.register("MyODE", MyODE)
```

---

## Known issues and limitations

- `var_names` must be set explicitly on the trainer instance (`trainer.var_names = ["prey", "predator"]`) or inferred from `target_cols` — no automatic wiring yet.
- `_phase_overlay_history` grows unboundedly. Use `deque(maxlen=N)` for long runs:

```python
from collections import deque
trainer._phase_overlay_history = deque(maxlen=20)
```

- `physics_loss_residual` uses `torch.autograd.grad` and must **not** be called inside `torch.no_grad()`.