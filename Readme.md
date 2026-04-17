# PINNs Framework

This project implements a PINN (Physics-Informed Neural Networks) framework in PyTorch for solving ordinary differential equations (ODEs) and ODE systems.

The neural network is trained directly on a loss function derived from the governing physical equation, reducing dependence on purely simulated data.

## Project Structure

```text
PINNSFramework/
│
├── pyproject.toml           # Project metadata and dependencies
├── main.py                  # Minimal entry script with a startup message
├── Readme.md                # Project documentation
├── src/                     # Main source code
│   ├── core/                # Checkpoint management and internal utilities
│   ├── model_repository/    # PINN model definitions
│   ├── ode_repository/      # Supported ODE definitions
│   ├── odes/                # Additional ODE-related modules
│   ├── visualizers/         # Visualization helpers
│   ├── data_models.py       # Configuration models and enums
│   ├── inference_runner.py  # Load trained model and run inference
│   ├── logger.py            # Logger setup
│   ├── run.py               # Primary training script
│   └── trainer_runner.py    # Training loop and PINN logic
├── runs/                    # Training outputs, checkpoints, TensorBoard logs
└── tests/                   # Unit tests
```

## Requirements

- Python 3.10+
- PyTorch 2.x
- numpy
- matplotlib
- scipy
- tensorboard
- pydantic
- structlog

> Dependencies are declared in `pyproject.toml`.

### CUDA / GPU Support

GPU support is optional but recommended for faster training. If you want to use CUDA, install a PyTorch build matching your CUDA version. For example, for CUDA 12.1:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you do not have a GPU or prefer CPU-only execution, install the CPU version of PyTorch:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

3. Install project dependencies:

```powershell
python -m pip install -e .
```

If you prefer a manual install without `pip install -e .`:

```powershell
python -m pip install matplotlib numpy scipy pydantic structlog tensorboard torch torchvision torchaudio
```

4. See Results 

Use tensorboard 

```powershell
tensorboard --logdir=runs\Lotka-Voltera
```

## CLI Usage

The `src/cli.py` file provides a command-line interface for:

- `train`: start a training run with `Trainer`
- `infer`: load a saved model and run inference
- `generate`: generate simulation data for an ODE

### Examples

Lotka-Volterra training:

```powershell
python src/cli/app.py train  ` 
  --ode lotka_volterra  `
  --model-dimension 2  `
  --initial-conditions 10 10  `
  --ode-params alpha=0.5 beta=1 gamma=1 delta=1
```

CFAST training:

```powershell
python src/cli.py train --ode CFAST --initial-conditions 101325 293 293 0 --model-dimension 4 --grid-size 800 --t-span 0 1 \
  --ode-params total_volume=250
```

Inference:

```powershell
python src/cli.py train `
  --ode CFAST `
  --initial-conditions 101325 293 293 0 `
  --model-dimension 4 `
  --grid-size 800 `
  --t-span 0 1 `
  --ode-params total_volume=250 `
```

Data generation:

```powershell
python src/cli.py generate --ode Lotka-Voltera --output-file data/lotka_volterra.parquet --n-sims 50 --n-steps 200
```

### ODE parameters

ODE parameters are provided as `key=value` pairs via `--ode-params`.

Examples:

- Lotka-Volterra: `alpha=0.6666667 beta=1.3333333 gamma=1 delta=1`
- CFAST: `total_volume=250`

## Training

The main training entry point is `src/run.py`:

```powershell
python src/run.py
```

This script creates a `Trainer` instance with the configured ODE case and starts training.

By default, `src/run.py` uses the `CFAST` case. To run Lotka-Volterra, change the `case` variable in `src/run.py` to:

```python
case = AvailablesODE.LOTKA_VOLTERA
```

Then run:

```powershell
python src/run.py
```

Training outputs are saved under `runs/` in folders such as `Lotka-Voltera/` or `CFAST/`, with subdirectories for `experiment_<date>`, `save/`, and `tensorboard_logs/`.

### Lotka-Volterra Example

The Lotka-Volterra case uses the following configuration in `src/run.py`:

- `initial_conditions=[10.0, 10.0]`
- `grid_size=800`
- `model_dimension=2`
- `epochs=200`
- `l_r=1e-3`
- `alpha=2/3`, `beta=4/3`, `gamma=1`, `delta=1`

This produces a 2-dimensional PINN that predicts the predator-prey dynamics of the Lotka-Volterra system.

### CFAST Example

The CFAST case is the default configuration in `src/run.py` and uses:

- `initial_conditions=[101325.0, 293.0, 293.0, 0.0]`
- `grid_size=800`
- `model_dimension=4`
- `epochs=4000`
- `l_r=1e-3`
- `total_volume=250`

This produces a 4-dimensional PINN for the CFAST model with pressure, upper temperature, lower temperature, and upper volume as the predicted state variables.

## Inference

To load a trained model and run inference:

```powershell
python -m src.inference_runner
```

Or from a Python script:

```python
from pathlib import Path
from src.inference_runner import InferenceRunner

runner = InferenceRunner.from_config(Path('runs/Lotka-Voltera/experiment_2026-04-16_15-13-10'))
runner.load()
runner.plot()
```

## Supported Options

- `Lotka-Voltera`
- `CFAST`
- Base PINN model: `BasicPINNS`

Training and ODE configuration parameters are defined in `src/data_models.py`.

## Visualization

TensorBoard logs are written inside each experiment folder under `tensorboard_logs`.

To view training curves:

```powershell
tensorboard --logdir=runs
```

Then open `http://localhost:6006` in your browser.

## Notes

- The actual training script is `src/run.py`.
- Checkpoints are managed by `src/core/checkpoint_manager.py`.

---

## Future Work

- Add more ODE and PDE cases
- Improve visualization utilities
- Add more unit tests

