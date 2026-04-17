# src/cli.py

import typer
import torch
from typing import List, Optional
from pathlib import Path

from src.cli.models import TrainConfig, InferConfig, GenerateConfig
from src.core.trainer_runner import Trainer
from src.data_models import TrainingConfig, ODEExperiment, AvailablesODE, AvailablesAIModel, ODESConfig, DataConfig

app = typer.Typer()


# ---------- Utils ----------

def parse_kv(value: Optional[List[str]]) -> dict:
    params = {}
    if value:
        for item in value:
            key, val = item.split("=")
            params[key] = float(val)
    return params


def auto_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"




# ---------- TRAIN ----------

@app.command()
def train(
    ode: str = typer.Option(...),
    model: str = typer.Option("basic_pinn"),

    lr: float = 1e-3,
    epochs: int = 2000,
    grid_size: int = 200,

    model_dimension: int = typer.Option(2),
    initial_conditions: List[float] = typer.Option([1.0, 10.0]),

    t_span: List[float] = typer.Option([0.0, 10.0]),

    ode_params: Optional[List[str]] = typer.Option(None),
    lambda_ode: float = 1.0,

    output_dir: str = "runs",
    device: Optional[str] = None,
):
    config = TrainConfig(
        ode=ode,
        model=model,
        lr=lr,
        epochs=epochs,
        grid_size=grid_size,
        model_dimension=model_dimension,
        initial_conditions=initial_conditions,
        t_span=t_span,
        ode_params=parse_kv(ode_params),
        output_dir=output_dir,
        device=auto_device(device),
        lambda_ode=lambda_ode
    )

    config.validate()

    typer.echo(f"Training {config.ode} on {config.device}")
    typer.echo(config.model_dump())

    ode_experiment_config = ODEExperiment(
            ode_name=config.ode,
            initial_conditions=config.initial_conditions,       
            model_dimension=config.model_dimension,
            input_cols=None,            
            target_cols=None,
            ode_config=ODESConfig(      
                parameters=config.ode_params,
                ode_name=AvailablesODE(config.ode),
                grid_size=config.grid_size,
                t_span=tuple(config.t_span),
                lambda_ode=config.lambda_ode
            ),  
            data_config=DataConfig(
                data_folder=Path(config.output_dir) / config.ode / "data",
                lambda_data=1.0,
                batch_size=64,          
                shuffle=True
            )
    )

    trainer = Trainer(
        training_config=TrainingConfig(l_r=config.lr, epochs=config.epochs),
        ode_experiment_config=ode_experiment_config,
        device=config.device,
        output_folder_path=Path(config.output_dir) / config.ode,                                   
        )
    trainer.run()


# ---------- INFER ----------

@app.command()
def infer(
    experiment_dir: str = typer.Option(...),
    plot: bool = False,
    save_plot: bool = False,
    device: Optional[str] = None,
):
    config = InferConfig(
        experiment_dir=experiment_dir,
        plot=plot,
        save_plot=save_plot,
        device=auto_device(device),
    )

    typer.echo(f"Inference from {config.experiment_dir}")
    typer.echo(config.model_dump())

    # infer_pipeline(config)


# ---------- GENERATE ----------

@app.command()
def generate(
    ode: str = typer.Option(...),

    ode_params: Optional[List[str]] = typer.Option(None),

    output_file: str = "data/generated_dataset.parquet",
    n_sims: int = 50,
    t_max: float = 10.0,
    n_steps: int = 200,
    seed: int = 42,
):
    config = GenerateConfig(
        ode=ode,
        ode_params=parse_kv(ode_params),
        output_file=output_file,
        n_sims=n_sims,
        t_max=t_max,
        n_steps=n_steps,
        seed=seed,
    )

    typer.echo(f"Generating dataset for {config.ode}")
    typer.echo(config.model_dump())

    # generate_pipeline(config)


if __name__ == "__main__":
    app()