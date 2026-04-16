import torch 
from scipy.integrate import solve_ivp
import numpy as np 
from pydantic import BaseModel

from src.ode_repository.ode_base import BaseODE
import matplotlib.pyplot as plt
from io import BytesIO


class ParamsLotkaVoltera(BaseModel):
    alpha: float = 1.
    beta: float = 2.
    delta: float = 1.
    gamma: float = 2.

class LotkaVoltera(BaseODE):
    def __init__(self, params: ParamsLotkaVoltera):
        self.name = "Lotka Volterra"
        self.params = params

    def update_params(self, new_params: ParamsLotkaVoltera):
        self.params = new_params

    # ---------- Core dynamics (factorisée) ----------
    def _dynamics(self, x, t=None):
        dx = self.params.alpha * x[:,0] - self.params.beta * x[:,0] * x[:,1]
        dy = self.params.delta * x[:,0] *  x[:,1]  - self.params.gamma *  x[:,1]
        return dx, dy

    # ---------- Torch version ----------
    def torch_ode(self, x: torch.Tensor) -> torch.Tensor:
        dx, dy = self._dynamics(x)
        return torch.stack((dx, dy), dim=1)

    # ---------- Numpy / SciPy version ----------
    def _dynamics_numpy(self, t, x):
        dx = self.params.alpha * x[0] - self.params.beta * x[0] * x[1]
        dy = self.params.delta * x[0] * x[1] - self.params.gamma * x[1]
        return [dx, dy]

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


    def log_trajectory_plot(self, t_true,y_true, y_pred):
        fig, ax = plt.subplots()

        ax.plot(t_true, y_true[:,0], label="Proies (SciPy)")
        ax.plot(t_true, y_true[:,1], label="Prédateurs (SciPy)")
        ax.plot(t_true, y_pred[:, 0], '--', label="Proies (NN)")
        ax.plot(t_true, y_pred[:, 1], '--', label="Prédateurs (NN)")

        ax.legend()
        ax.set_xlabel("t")
        ax.set_ylabel("Population")

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        img = torch.tensor(np.array(plt.imread(buf))).permute(2, 0, 1)

        plt.close(fig)
        return img
       

    # ---------- Simple visualization ----------
    def log_trajectory_phase_space_plot(self,y_true, y_pred):
        fig, ax = plt.subplots()

        ax.plot(y_true[:, 0], y_true[:, 1], label="Proies / Prédateurs (SciPy)")
        ax.plot(y_pred[:, 0], y_pred[:, 1], '--', label="Proies / Prédateurs (NN)")

        ax.legend()
        ax.set_xlabel("Proies")
        ax.set_ylabel("Prédateurs")

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        img = torch.tensor(np.array(plt.imread(buf))).permute(2, 0, 1)

        plt.close(fig)
        return img