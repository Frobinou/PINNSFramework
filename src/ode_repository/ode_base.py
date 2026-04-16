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
    def torch_ode(self,x: torch.Tensor, params: BaseModel) -> torch.Tensor:
        raise NotImplementedError

    # ---------- Numpy / SciPy version ----------
    def numpy_ode(self,t, x, params: BaseModel):
        raise NotImplementedError

    # ---------- Validation ----------
    
    def validation(self, t_span: tuple, x0: list):
        t_eval = np.linspace(*t_span, 200)

        sol = solve_ivp(
            fun=self._dynamics_numpy,
            t_span=t_span,
            y0=x0,
            t_eval=t_eval
        )

        return sol