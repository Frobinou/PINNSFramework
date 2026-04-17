import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

cwd = str(Path.cwd())
sys.path.append(cwd)

import numpy as np
from src.core.trainer_runner import Trainer
from src.data_models import TrainingConfig, ODEExperiment, AvailablesODE, AvailablesAIModel, ODESConfig

# CLI entrypoint follows

from src.core.inference_runner import InferenceRunner
from src.odes.data_generator.ode_data_generator import BaseODEDataGenerator


def parse_value(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def parse_key_value_pairs(raw_pairs: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not raw_pairs:
        return result

    for raw in raw_pairs:
        if '=' not in raw:
            raise ValueError(f"Invalid parameter format: '{raw}'. Expected key=value")
        key, value = raw.split('=', 1)
        result[key] = parse_value(value)
    return result


def build_ode_parameters(ode_name: AvailablesODE, params: dict[str, Any]) -> Any:
    if ode_name == AvailablesODE.LOTKA_VOLTERA:
        from src.odes.ode_repository.ode_lotka_voltera import ParamsLotkaVoltera

        return ParamsLotkaVoltera(**params)

    if ode_name == AvailablesODE.CFAST:
        from src.odes.ode_repository.ode_cfast import ParamsCFAST

        return ParamsCFAST(**params)

    raise ValueError(f"Unsupported ODE '{ode_name}'")


def build_default_generator(ode_name: AvailablesODE, ode_params: dict[str, Any]):
    if ode_name == AvailablesODE.LOTKA_VOLTERA:
        from src.odes.ode_repository.ode_lotka_voltera import LotkaVoltera, ParamsLotkaVoltera

        params = ParamsLotkaVoltera(**ode_params)
        ode = LotkaVoltera(params=params)

        def sample_x0():
            return np.array([np.random.uniform(5, 15), np.random.uniform(3, 10)])

        def sample_params():
            return {
                "alpha": np.random.uniform(0.5, 2.0),
                "beta": np.random.uniform(0.01, 0.5),
                "delta": np.random.uniform(0.01, 0.5),
                "gamma": np.random.uniform(0.5, 2.0),
            }

        return ode, params, sample_x0, sample_params

    if ode_name == AvailablesODE.CFAST:
        from src.odes.ode_repository.ode_cfast import ODECFAST, ParamsCFAST

        params = ParamsCFAST(**ode_params)
        ode = ODECFAST(params=params)

        def sample_x0():
            return np.array([101325.0, 293.0, 293.0, 0.0])

        def sample_params():
            return {
                "total_volume": params.total_volume,
                "gamma": params.gamma,
                "cp": params.cp,
                "combustion_heat": params.combustion_heat,
                "combustion_speed": params.combustion_speed,
                "outside_temperature": params.outside_temperature,
                "Q": params.Q,
                "R": params.R,
            }

        return ode, params, sample_x0, sample_params

    raise ValueError(f"Cannot create data generator for ODE '{ode_name}'")


def create_ode_config(args: Any) -> ODESConfig:
    return ODESConfig(
        ode_name=AvailablesODE(args.ode),
        initial_conditions=args.initial_conditions,
        model_dimension=args.model_dimension,
        grid_size=args.grid_size,
        t_span=tuple(args.t_span)
    )


def train_command(args: Any) -> None:
    ode_params = parse_key_value_pairs(args.ode_params)
    ode_parameters = build_ode_parameters(AvailablesODE(args.ode), ode_params)

    trainer = Trainer(
        training_config=TrainingConfig(l_r=args.lr, epochs=args.epochs),
                                        ode_experiment_config=ODEExperiment(
                                            ode_config=create_ode_config(args),
                                            data_config=None
                                        ),
        ode_parameters=ode_parameters,
        device=args.device,
        output_folder_path=Path(args.output_dir) / args.ode,
        model_name=AvailablesAIModel(args.model),
    )
    trainer.run()


def infer_command(args: Any) -> None:
    runner = InferenceRunner.from_config(experiment_dir=Path(args.experiment_dir))
    if args.device:
        runner.device = args.device

    runner.load()
    if args.plot:
        runner.plot(save=args.save_plot)
    else:
        t, pred = runner.predict()
        print(f"Predictions shape: {pred.shape}")


def generate_command(args: Any) -> None:
    ode_name = AvailablesODE(args.ode)
    ode_params = parse_key_value_pairs(args.ode_params)
    ode, _, x0_sampler, param_sampler = build_default_generator(ode_name, ode_params)

    generator = BaseODEDataGenerator(params=ode_params, ode=ode)
    generator.generate_and_save(
        file_path=Path(args.output_file),
        n_sims=args.n_sims,
        x0_sampler=x0_sampler,
        param_sampler=param_sampler,
        t_max=args.t_max,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    print(f"Saved generated data to {args.output_file}")



def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="PINNSFramework command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a PINN model")
    train_parser.add_argument("--ode", choices=[item.value for item in AvailablesODE], required=True)
    train_parser.add_argument("--model", choices=[item.value for item in AvailablesAIModel], default=AvailablesAIModel.BASIC_PINN.value)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--epochs", type=int, default=2000)
    train_parser.add_argument("--grid-size", type=int, default=200)
    train_parser.add_argument("--model-dimension", type=int, default=2)
    train_parser.add_argument("--initial-conditions", type=float, nargs="+", required=True)
    train_parser.add_argument("--t-span", type=float, nargs=2, default=(0.0, 10.0))
    train_parser.add_argument("--ode-params", nargs="*", default=[], help="ODE parameters as key=value")
    train_parser.add_argument("--output-dir", default="runs", help="Base output directory for experiments")
    train_parser.add_argument("--device", default="", help="Torch device override")
    train_parser.set_defaults(func=train_command)

    infer_parser = subparsers.add_parser("infer", help="Run inference from a trained experiment")
    infer_parser.add_argument("--experiment-dir", required=True, help="Path to the experiment folder")
    infer_parser.add_argument("--plot", action="store_true", help="Generate a plot for the inference results")
    infer_parser.add_argument("--save-plot", action="store_true", help="Save the inference plot to disk")
    infer_parser.add_argument("--device", default="", help="Torch device override")
    infer_parser.set_defaults(func=infer_command)

    generate_parser = subparsers.add_parser("generate", help="Generate simulation dataset")
    generate_parser.add_argument("--ode", choices=[item.value for item in AvailablesODE], required=True)
    generate_parser.add_argument("--ode-params", nargs="*", default=[], help="ODE parameters as key=value")
    generate_parser.add_argument("--output-file", default="data/generated_dataset.parquet", help="Output dataset file path")
    generate_parser.add_argument("--n-sims", type=int, default=50)
    generate_parser.add_argument("--t-max", type=float, default=10.0)
    generate_parser.add_argument("--n-steps", type=int, default=200)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.set_defaults(func=generate_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
