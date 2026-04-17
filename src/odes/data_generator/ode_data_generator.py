from abc import ABC
import os
import numpy as np
import pandas as pd
from src.odes.ode_repository.ode_base import BaseODE
from pathlib import Path

class ODEDataGenerator(ABC):
    """
    Abstract interface for ODE systems.
    Designed to generalize Lotka-Volterra, CFAST, and other dynamical systems.
    """

    def __init__(self, params: dict, 
                 ode: BaseODE, 
                 target_cols: list[str], 
                 input_cols: list[str],
                 t_span: tuple[float, float] = (0, 10), n_steps: int = 200):
        self.params = params
        self.ode: BaseODE = ode
        self.target_cols = target_cols
        self.input_cols = input_cols
        self.t_span = t_span
        self.n_steps = n_steps


    def simulate(self, x0) -> pd.DataFrame:
        sol = self.ode.simulate(
            t_span=self.t_span,
            x0=x0,
            nb_points=self.n_steps
        )   

        # Build dataframe from solution
        df = pd.DataFrame(sol.y.T, columns= self.target_cols)
        df["t"] = sol.t

        # Attach parameters to dataset
        for k, v in self.params.items():
            print(k,v)
            df[k] = v

        print(df)
        return df

    # -----------------------------
    # Multi-simulation dataset generator
    # -----------------------------
    def generate_dataset(self,
                         n_sims=100,
                         x0_sampler=None,
                         param_sampler=None,
                         seed=42):

        np.random.seed(seed)
        data = []

        for i in range(n_sims):

            # Sample parameters if sampler is provided
            if param_sampler is not None:
                self.params = param_sampler()

            # Sample initial condition
            if x0_sampler is not None:
                x0 = x0_sampler() if callable(x0_sampler) else x0_sampler[i]
            else:
                x0 = np.ones(2)

            df = self.simulate(x0)
            df["run_id"] = i

            data.append(df)

        return pd.concat(data, ignore_index=True)


    def generate_and_save(self, file_path:Path, **kwargs):
        if not file_path.exists():
            os.makedirs(file_path.parent, exist_ok=True)

        df = self.generate_dataset(**kwargs)
        df.to_parquet(file_path, index=False)
        return df
    

    # -----------------------------
    # PINN-ready format conversion
    # -----------------------------
    @staticmethod
    def to_pinn_format(df, state_cols):
        """
        Converts dataframe to PINN input-output format.

        Returns:
            x: input features (t + parameters)
            y: state variables
        """

        x = df[["t"] +
               [k for k in df.columns if k in state_cols] +
               [k for k in df.columns if k not in state_cols + ["t", "run_id"]]
              ].values

        y = df[state_cols].values

        return x.astype(np.float32), y.astype(np.float32)