# src/core/bootstrap.py

def bootstrap_registry() -> None:
    """Import all registered modules to trigger decorator execution.

    Must be called once at application startup, before any
    ``REGISTRY.xxx.get()`` or ``REGISTRY.xxx.build()`` call.

    Example:
```python
        from src.core.bootstrap import bootstrap_registry
        bootstrap_registry()
        print(REGISTRY)
        # GlobalRegistry:
        #   odes: ['lotka_voltera', 'van_der_pol']
        #   models: ['basic_pinn']
        #   losses: ['PINN_LOSS']
        #   data_loaders: ['parquet']
```
    """
    # ── ODEs ──────────────────────────────────────────────────────────────────
    import src.repositories.odes.ode_repository.ode_lotka_voltera  # noqa: F401
    import src.repositories.odes.ode_repository.ode_cfast          # noqa: F401

    # ── Models ────────────────────────────────────────────────────────────────
    import src.repositories.models.model_PINN                       # noqa: F401

    # ── Losses ────────────────────────────────────────────────────────────────
    import src.repositories.losses.pinn_losses                      # noqa: F401

    # ── DataLoaders ───────────────────────────────────────────────────────────
    import src.repositories.data_loader.parquet_dataloader          # noqa: F401