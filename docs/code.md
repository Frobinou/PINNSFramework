# About code

## Core

### Callback

A **Callback** is a hook system that lets you inject custom logic into the training loop
without modifying the `Trainer` itself.

At each stage of training — before the run starts, before and after each epoch, and when
the run ends — the `Trainer` calls the corresponding method on every registered callback.
This keeps the training loop minimal and unopinionated while allowing arbitrarily complex
side-effects: logging metrics to TensorBoard, saving top-k checkpoints, halting early when
validation loss stagnates, etc.

**Built-in callbacks**

| Callback | Trigger | Effect |
|---|---|---|
| `TensorBoardCallback` | `on_epoch_end` | Writes loss scalars to a TensorBoard event file |
| `CheckpointCallback` | `on_epoch_end` | Saves model & optimizer state if loss ranks in top-k |
| `EarlyStoppingCallback` | `on_epoch_end` | Raises a stop signal when val loss has not improved for `patience` epochs |

**Adding a custom callback**

Subclass `Callback` and override only the methods you need:

```python
from src.core.callback.base_callback import Callback

class PrintLossCallback(Callback):
    def on_epoch_end(self, trainer, epoch: int) -> None:
        print(f"[{epoch}] loss = {trainer.last_loss:.6f}")
```

Then register it before calling `.fit()`:

```python
trainer.callbacks.append(PrintLossCallback())
```

::: core.callback.base_callback.Callback

---

### Evaluator

An **Evaluator** measures model performance at the end of each epoch, independently
from the training loss.

Where the loss drives gradient updates, evaluators are read-only observers — they run
under `torch.no_grad()` and report metrics without interfering with the optimisation.
Each evaluator receives the full `Trainer` instance, giving it access to the model,
device, and any state it needs to compute its metric.

**Built-in evaluators**

| Evaluator | Metric | Data source |
|---|---|---|
| `MSEEvaluator` | Mean Squared Error on held-out data | Validation `DataLoader` |
| `ODEEvaluator` | ODE residual on the collocation grid | Collocation points `t` |

**Adding a custom evaluator**

Subclass `Evaluator` and implement `run()`:

```python
from src.core.evaluator.base_evaluator import Evaluator

class MaxErrorEvaluator(Evaluator):
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def run(self, trainer) -> dict:
        max_err = 0.0
        for batch in self.dataloader.get_val_loader():
            x = batch["x"].to(trainer.device)
            y = batch["y"].to(trainer.device)
            with torch.no_grad():
                err = (trainer.model(x) - y).abs().max().item()
            max_err = max(max_err, err)
        return {"max_error": max_err}
```

Then register it before calling `.fit()`:

```python
trainer.evaluators.append(MaxErrorEvaluator(make_dataloader(data_config)))
```

**Difference with Callbacks**

| | Callback | Evaluator |
|---|---|---|
| Purpose | Side-effects (logging, saving, stopping) | Metric computation |
| Return value | `None` | `dict[str, float]` |
| Gradient context | Unrestricted | Always `no_grad()` |

::: core.evaluator.base_evaluator.Evaluator

---
### Bootstrap

The **bootstrap** is a single startup routine that makes all registered components
available in the :data:`~src.core.registry.REGISTRY` before any training code runs.

**Why it is needed**

Python only executes a module's top-level code — including `@REGISTRY.xxx.register(...)`
decorators — when that module is **imported**. If a module is never imported, its
components are never registered, and any `REGISTRY.xxx.get()` call will raise a
`KeyError` at runtime.

`bootstrap_registry()` solves this by importing every registrable module in one place,
guaranteeing the registry is fully populated before `build_trainer()` or
`make_dataloader()` are called.

**Without bootstrap**

```python
REGISTRY.models.get("basic_pinn")
# KeyError: [models] Unknown key: 'basic_pinn'. Available: []
```

**With bootstrap**

```python
from src.core.bootstrap import bootstrap_registry
bootstrap_registry()

REGISTRY.models.get("basic_pinn")
# <class 'src.repositories.models.model_PINN.PINN'>
```

**When to call it**

Call `bootstrap_registry()` once, at the very top of your entry point, before any
registry lookup:

```python
# main.py
from src.core.bootstrap import bootstrap_registry
bootstrap_registry()

from src.core.factory import build_trainer, make_dataloader
...
```

**Adding a new component**

When you create a new ODE, model, loss, or dataloader, register it in two steps:

1. Decorate the class:
```python
# src/repositories/odes/ode_repository/ode_my_ode.py
@REGISTRY.odes.register("my_ode")
class MyODE:
    ...
```

2. Add the import to `bootstrap.py`:
```python
import src.repositories.odes.ode_repository.ode_my_ode  # noqa: F401
```

The `# noqa: F401` suppresses the *"imported but unused"* linter warning — the import
is intentional, its sole purpose being to trigger the decorator as a side effect.

::: core.bootstrap

---
### Registry

The **Registry** is a lightweight service locator that maps string keys to classes,
letting you swap any component — ODE, model, loss, dataloader — from a config file
without changing a single line of training code.

**Architecture**

The global :data:`~src.core.registry.REGISTRY` is a `GlobalRegistry` instance
exposing four independent sub-registries:

```
REGISTRY
├── odes          → ODE classes          (e.g. "lotka_voltera")
├── models        → Model classes        (e.g. "basic_pinn")
├── losses        → Loss classes         (e.g. "PINN_LOSS")
└── data_loaders  → DataLoader classes   (e.g. "parquet")
```

Each sub-registry is a `Registry` instance that stores a `{ name → class }` mapping
and validates keys at both registration and lookup time.

**Registering a component**

Use the `@REGISTRY.xxx.register("key")` decorator on the class definition:

```python
from src.core.registry import REGISTRY

@REGISTRY.models.register("basic_pinn")
class PINN(torch.nn.Module):
    ...
```

Optional metadata can be attached for documentation or introspection:

```python
@REGISTRY.models.register("basic_pinn", description="Vanilla PINN", version="1.0")
class PINN(torch.nn.Module):
    ...
```

**Looking up and instantiating**

Retrieve a class with `.get()` and instantiate it manually:

```python
cls = REGISTRY.models.get("basic_pinn")
model = cls(output_dim=2)
```

Or instantiate directly with `.build()`:

```python
model = REGISTRY.models.build("basic_pinn", output_dim=2)
```

**Introspection**

```python
# List all registered keys
REGISTRY.models.list()
# ['basic_pinn']

# Check if a key exists
"basic_pinn" in REGISTRY.models
# True

# Full metadata for all entries
REGISTRY.models.info()
# {'basic_pinn': {'class': 'PINN', 'description': 'Vanilla PINN', 'version': '1.0'}}

# Overview of the entire registry
print(REGISTRY)
# GlobalRegistry:
#   odes: ['lotka_voltera', 'van_der_pol']
#   models: ['basic_pinn']
#   losses: ['PINN_LOSS']
#   data_loaders: ['parquet']
```

**Error handling**

Registering a duplicate key raises immediately:

```python
@REGISTRY.models.register("basic_pinn")  # already taken
class AnotherModel:
    ...
# ValueError: [models] 'basic_pinn' is already registered.
# Existing: PINN, New: AnotherModel
```

Looking up an unknown key raises with available options:

```python
REGISTRY.models.get("unknown")
# KeyError: [models] Unknown key: 'unknown'. Available: ['basic_pinn']
```

::: core.registry.Registry
::: core.registry.GlobalRegistry

### Schemas

**Schemas** are [Pydantic](https://docs.pydantic.dev) models that define, validate, and
serialise every configuration object in the experiment. They act as the single source of
truth for all hyperparameters — from ODE physics to training schedule — and can be dumped
to JSON for full experiment reproducibility.

**Overview**

```
ExperimentConfig
├── ODESConfig        → ODE physics & simulation grid
├── DataConfig        → DataLoader construction
├── PhysicsWeights    → Loss term weights
└── TrainingConfig    → Training hyperparameters
```

**Validation**

All fields are validated at instantiation. Invalid values raise a `ValidationError`
immediately, before any training code runs:

```python
DataConfig(
    parquet_path=Path("data/lotka_volterra.parquet"),
    input_cols=["t"],
    target_cols=["prey", "predator"],
    batch_size=-1,  # ValidationError: batch_size must be > 0
)
```

`ODESConfig` additionally checks that `initial_conditions` matches `dimension`:

```python
ODESConfig(
    ode_name=AvailablesODE.LOTKA_VOLTERA,
    parameters={"alpha": 0.67, "beta": 1.33, "delta": 1.0, "gamma": 1.0},
    dimension=2,
    initial_conditions=[1.0],  # ValidationError: got 1 element, expected 2
)
```

**Serialisation**

Any schema — or the top-level `ExperimentConfig` — can be dumped to JSON and reloaded
exactly, enabling full experiment reproducibility:

```python
# Dump to file
path = Path("runs/Lotka-Voltera/config.json")
path.write_text(experiment.model_dump_json(indent=4))

# Reload from file
experiment = ExperimentConfig.model_validate_json(path.read_text())
```

The `CheckpointManager` calls `save_config()` automatically at the start of each run,
so every checkpoint folder contains its own `config.json`.

**Full example**

```python
from src.core.schemas import (
    ExperimentConfig,
    ODESConfig,
    DataConfig,
    PhysicsWeights,
    TrainingConfig,
    AvailablesODE,
)

experiment = ExperimentConfig(
    ode=ODESConfig(
        ode_name=AvailablesODE.LOTKA_VOLTERA,
        parameters={"alpha": 0.67, "beta": 1.33, "delta": 1.0, "gamma": 1.0},
        t_span=(0.0, 50.0),
        grid_size=2000,
        dimension=2,
        initial_conditions=[1.0, 1.0],
    ),
    data=DataConfig(
        parquet_path=Path("data/lotka_volterra.parquet"),
        input_cols=["t"],
        target_cols=["prey", "predator"],
        batch_size=64,
    ),
    physics=PhysicsWeights(lambda_ode=1.0, lambda_data=1.0),
    training=TrainingConfig(epochs=2000, lr=1e-3),
)
```

::: core.schemas.ODESConfig
::: core.schemas.DataConfig
::: core.schemas.PhysicsWeights
::: core.schemas.TrainingConfig
::: core.schemas.ExperimentConfig