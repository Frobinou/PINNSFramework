import torch
from typing import Literal


from src.core.registry import REGISTRY
from src.core.schemas import DataConfig
from src.core.trainer import Trainer
from src.core.schemas import (
    ODESConfig,
    TrainingConfig,
    PhysicsWeights
)

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