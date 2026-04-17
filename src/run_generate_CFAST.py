
import sys
from pathlib import Path
cwd = str(Path.cwd())
sys.path.append(cwd)

from pathlib import Path
import numpy as np
from typing import Any
from src.odes.data_generator.ode_data_generator import ODEDataGenerator



if __name__ == "__main__":
    from src.odes.ode_repository.ode_cfast import ODECFAST, ParamsCFAST

    # Instantiate model
    params = ParamsCFAST(total_volume=1000.0)
    model_ode = ODECFAST(params=params)

    generator = ODEDataGenerator(params=params.model_dump(mode='python'), 
                                 ode=model_ode,
                                 target_cols=["p", "T_u", "T_l", "V_u"],
                                 input_cols=["t", "gamma", "total_volume", "cp"],
                                 t_span=(0, 10),
                                 n_steps=500
                        )

    # Generate dataset
    df = generator.generate_and_save(
        file_path=Path("data/cfast.parquet"),
        n_sims=1,
        x0_sampler=[[10100, 293, 293,0.1]], # Not really sampling, but could be replaced with sample_x0 for more variability
        param_sampler=None
    )