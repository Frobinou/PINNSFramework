import torch
from typing import Literal
from pathlib import Path

from src.core.registry import REGISTRY
from src.core.schemas import DataConfig
from src.core.trainer import Trainer
from src.core.schemas import (
    ODESConfig,
    TrainingConfig,
    PhysicsWeights
)
from src.core.experiment_io import load_experiment

def build_trainer(
    ode_config: ODESConfig,
    loss_config:PhysicsWeights,
    training_config:TrainingConfig,
    device: Literal['cpu', 'cuda'] = "cpu",
) -> Trainer:
    """Build a fully configured :class:`~src.core.trainer.Trainer` from config objects.

    Resolves each component (ODE, model, loss, optimizer) from the global
    :data:`~src.core.registry.REGISTRY` and assembles them into a ready-to-use
    trainer. No manual instantiation is required in the calling code.

    Args:
        ode_config: ODE configuration. Must expose:

            - ``ode_name`` *(str)*: registry key for the ODE class.
            - ``parameters`` *(dict)*: passed as ``params=`` to the ODE constructor.
            - ``dimension`` *(int)*: dimension of the ODE system, used to build the model.


        loss_config: Loss configuration. Must expose:

            - ``name`` *(str)*: registry key for the loss class.
            - ``lambda_ode`` *(float)*: weight for the physics residual term.
            - ``lambda_data`` *(float)*: weight for the supervised data term.

        training_config: Training hyperparameters. Must expose:

            - ``l_r`` *(float)*: learning rate passed to ``torch.optim.Adam``.
            - model_name *(str)*: registry key for the model class.

        device: Target device for the trainer (``"cpu"`` or ``"cuda"``).
            Defaults to ``"cpu"``.

    Returns:
        A :class:`~src.core.trainer.Trainer` instance with model, optimizer,
        and loss function attached, ready to call ``.fit()``.

    Example:
```python
        trainer = build_trainer(
            ode_config=ode_config,
            loss_config=loss_config,
            training_config=training_config,
            device="cuda",
        )
        trainer.fit(dataloader=make_dataloader(data_config), epochs=2000)
```
    """
    ode = REGISTRY.odes.get(ode_config.ode_name)(params=ode_config.parameters)

    model = REGISTRY.models.get(training_config.model_name)(output_dim=ode_config.dimension)

    loss = REGISTRY.losses.get(loss_config.name)(
        ode=ode,
        lambda_ode=loss_config.lambda_ode,
        lambda_data=loss_config.lambda_data,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)

    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        device=device,
        t=torch.linspace(ode_config.t_span[0], ode_config.t_span[1], ode_config.grid_size)
    )


def make_dataloader(config: DataConfig):
    """Build a :class:`~torch.utils.data.DataLoader` from a :class:`~src.core.schemas.DataConfig`.

    Delegates construction to the ``"parquet"`` entry in
    :data:`~src.core.registry.REGISTRY`, which returns a
    :class:`~src.repositories.data_loader.parquet_dataloader.ParquetDataLoader`
    wrapping train / val / test splits.

    Note:
        For PINNs, ``input_cols`` typically contains only ``["t"]`` even if the
        underlying Parquet file contains additional columns (e.g. ODE parameters).
        The split ratios and seed are inherited from
        :class:`~src.core.schemas.DataConfig` defaults.

    Args:
        config: Data configuration object. Must expose:

            - ``data_path`` *(Path)*: path to the ``.parquet`` file.
            - ``input_cols`` *(list[str])*: feature columns fed to the model.
            - ``target_cols`` *(list[str])*: target columns the model must predict.
            - ``batch_size`` *(int)*: number of samples per batch.

    Returns:
        A :class:`~src.repositories.data_loader.parquet_dataloader.ParquetDataLoader`
        exposing ``get_train_loader()``, ``get_val_loader()``, and
        ``get_test_loader()``.

    Example:
```python
        data_config = DataConfig(
            data_path=Path("data/lotka_volterra.parquet"),
            input_cols=["t"],
            target_cols=["prey", "predator"],
            batch_size=64,
        )
        loader = make_dataloader(data_config)
        trainer.fit(dataloader=loader, epochs=2000)
```
    """
    return REGISTRY.data_loaders.build(
        name=config.type,
        data_path=config.data_path,
        input_cols=config.input_cols,
        target_cols=config.target_cols,
        batch_size=config.batch_size,
    )

def run_inference(
    trainer: Trainer,
    ode_config: ODESConfig,
    device: Literal['cpu', 'cuda'] = "cpu",
) -> torch.Tensor:
    """Run inference over the full t_span defined in ode_config.

    Sets the model to evaluation mode and disables gradient computation
    for efficiency.

    Args:
        trainer: Trained trainer instance with model attached.
        ode_config: ODE configuration exposing ``t_span`` and ``grid_size``.
        device: Target device (``"cpu"`` or ``"cuda"``). Defaults to ``"cpu"``.

    Returns:
        Tensor of shape ``(grid_size, output_dim)`` with model predictions.

    Example:
```python
        y_pred = run_inference(trainer, ode_config, device="cuda")
```
    """
    t = torch.linspace(
        ode_config.t_span[0],
        ode_config.t_span[1],
        ode_config.grid_size
    ).unsqueeze(1).to(device)

    trainer.model.eval()
    with torch.no_grad():
        y_pred = trainer.model(t)

    return y_pred


def run_inference_from_config(
    experiment_path: str | Path,
    checkpoint_path: str | Path,
    device: Literal['cpu', 'cuda'] = "cpu",
) -> torch.Tensor:
    """Rebuild a trainer from a saved experiment and run inference.

    Loads the experiment configuration from a JSON file, reconstructs
    the trainer, loads the model weights from a checkpoint, and runs
    inference over the full t_span.

    Args:
        experiment_path: Path to the experiment JSON file produced by
            :func:`~src.core.experiment_io.save_experiment`.
        checkpoint_path: Path to the model weights file (``.pt``) produced
            by the checkpoint callback.
        device: Target device (``"cpu"`` or ``"cuda"``). Defaults to ``"cpu"``.

    Returns:
        Tensor of shape ``(grid_size, output_dim)`` with model predictions.

    Example:
```python
        y_pred = run_inference_from_config(
            experiment_path="experiments/experiment_20260423_143512.json",
            checkpoint_path="checkpoints/best_model.pt",
            device="cuda",
        )
```
    """
    config = load_experiment(experiment_path)

    trainer = build_trainer(
        ode_config=config.ode,
        loss_config=config.physics_weights,
        training_config=config.training,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])

    return run_inference(trainer, config.ode, device)