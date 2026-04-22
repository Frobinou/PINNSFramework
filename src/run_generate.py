import sys
from pathlib import Path

cwd = str(Path.cwd())
sys.path.append(cwd)

from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from src.repositories.odes.data_generator.ode_data_generator import ODEDataGenerator


def sample_params():
    """Sample random model parameters."""
    return {
        "alpha": np.random.uniform(0.5, 2.0),
        "beta": np.random.uniform(0.01, 0.5),
        "delta": np.random.uniform(0.01, 0.5),
        "gamma": np.random.uniform(0.5, 2.0),
    }


def sample_x0():
    """Sample random initial conditions."""
    return np.array([np.random.uniform(5, 15), np.random.uniform(3, 10)])


()


if __name__ == "__main__":
    from src.repositories.odes.ode_repository.ode_lotka_voltera import LotkaVoltera, ParamsLotkaVoltera

    # Instantiate model
    params = ParamsLotkaVoltera(alpha=0.67, beta=1.3, delta=1.0, gamma=1.0)
    model_ode = LotkaVoltera(params=params)

    generator = ODEDataGenerator(
        params=params.model_dump(mode="python"),
        ode=model_ode,
        target_cols=["prey", "predator"],
        input_cols=["t", "alpha", "beta", "delta", "gamma"],
        t_span=(0, 50),
        n_steps=2000
    )
    # Generate dataset
    df = generator.generate_and_save(
        file_path=Path("data/lotka_volterra.parquet"),
        n_sims=1,
        x0_sampler=[
            [1.0, 1.0]
        ],  # Not really sampling, but could be replaced with sample_x0 for more variability
        param_sampler=None,
    )
    print(df)
    fig_traj = generator.plot_trajectories(df, max_runs=10)
    fig_traj.savefig("trajectories.png", dpi=150)

    fig_phase = generator.plot_phase_portrait(df, max_runs=10)
    plt.show()
