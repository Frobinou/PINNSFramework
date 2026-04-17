
import sys
from pathlib import Path
cwd = str(Path.cwd())
sys.path.append(cwd)

from pathlib import Path
import numpy as np
from typing import Any
from src.odes.data_generator.ode_data_generator import ODEDataGenerator

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
    return np.array([
        np.random.uniform(5, 15),
        np.random.uniform(3, 10)
    ])
()
    

if __name__ == "__main__":
    from src.odes.ode_repository.ode_lotka_voltera import LotkaVoltera, ParamsLotkaVoltera

    # Instantiate model
    params = ParamsLotkaVoltera(alpha=0.67, beta=1.3, delta=1., gamma=1.0)
    model_ode = LotkaVoltera(params=params)

    generator = ODEDataGenerator(params=params.model_dump(mode='python'), 
                                 ode=model_ode,
                                 target_cols=["prey", "predator"],
                                 input_cols=["t", "alpha", "beta", "delta", "gamma"],
                                 t_span=(0, 50)
                        )
    # Generate dataset
    df = generator.generate_and_save(
        file_path=Path("data/lotka_volterra.parquet"),
        n_sims=1,
        x0_sampler=[[10.0, 10.0]], # Not really sampling, but could be replaced with sample_x0 for more variability
        param_sampler=None
    )