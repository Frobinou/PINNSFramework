from abc import ABC
import torch
from pydantic import BaseModel

from scipy.integrate import solve_ivp
import numpy as np 

class BaseODE:
    def __init__(self, params: BaseModel):
        self.name = "Lotka Volterra"
        self.params: BaseModel = params

    def update_params(self, new_params: BaseModel):
        self.params = new_params

    # ---------- Core dynamics (factorisée) ----------
    def _dynamics(self,t, x, params):
        raise NotImplementedError

    # ---------- Torch version ----------
    def torch_ode(self, x: torch.Tensor) -> torch.Tensor:
        dx = self._dynamics(x.mT)
        return torch.stack(dx, dim=1)

    # ---------- Numpy / SciPy version ----------

    def _dynamics_numpy(self, t, x):
        return self._dynamics(x,t=t)
    # ---------- Validation ----------
    
    def validation(self, t_span: tuple, x0: list, nb_points:int):
        t_eval = np.linspace(*t_span, nb_points)

        sol = solve_ivp(
            fun=self._dynamics_numpy,
            t_span=t_span,
            y0=x0,
            t_eval=t_eval,
            #method="Radau",
            #max_step=0.01
        )

        return sol