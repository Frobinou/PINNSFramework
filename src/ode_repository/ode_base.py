from abc import ABC
import torch
from pydantic import BaseModel


class BaseODE:
    def __init__(self, params: BaseModel):
        self.name = "Lotka Volterra"
        self.params: BaseModel = params

    def update_params(self, new_params: BaseModel):
        self.params = new_params

    # ---------- Core dynamics (factorisée) ----------
    @staticmethod
    def _dynamics(t, x, params):
        raise NotImplementedError

    # ---------- Torch version ----------
    @staticmethod
    def torch_ode(x: torch.Tensor, params: BaseModel) -> torch.Tensor:
        raise NotImplementedError

    # ---------- Numpy / SciPy version ----------
    @staticmethod
    def numpy_ode(t, x, params: BaseModel):
        raise NotImplementedError

    # ---------- Validation ----------
    def validation(self, t_span: tuple, x0: list, params: BaseModel):
        raise NotImplementedError