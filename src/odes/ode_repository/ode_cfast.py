import torch 

from src.odes.ode_repository.ode_base import BaseODE
from src.odes.visualizers.base_visualizer import VisualizationMixin
from src.data_models import AvailablesODE

from pydantic import BaseModel

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

class ParamsCFAST(BaseModel):
    gamma: float = 1.4  # heat capacity ratio
    total_volume: float 
    cp: float = 1012 # J/kg/K
    combustion_heat: float = 24000 # in KJ/kg
    combustion_speed: float = 0.2 #in kg/s
    outside_temperature: float = 293#K, 20 degree
    Q: float = 0.44 / 1000 # MW
    R: float = 289.14 # Gaz constant J/kg/K

class ODECFAST(BaseODE):
    def __init__(self, params: ParamsCFAST):
        self.name = AvailablesODE.CFAST
        self.params: ParamsCFAST = params

    def update_params(self, new_params: BaseModel):
        self.params = new_params

    # ---------- Core dynamics (factorisée) ----------

    def _dynamics(self, x, t=None):
        """
         [p, T_u, T_l, V_u] 
        """
        q_dot_l, q_dot_u = self.compute_source_terms_heat_flux(x)
        dp = (self.params.gamma - 1)/self.params.total_volume * (q_dot_u + q_dot_l) # pressure equation

        # Compute law of perfect gaz m_i = M_i * P * V_i / (R * T_i)
        M = 29. # Hypothesis : only air
        m_l = M * x[0] * (self.params.total_volume - x[3]) / (self.params.R * x[2])
        m_u = M * x[0] * (x[3]) / (self.params.R * x[1])

        m_dot_l, m_dot_u = self.compute_source_terms_mass_flux()
        cst_u = self.params.cp * m_u
        dT_u = (1./cst_u) * (q_dot_u - self.params.cp * m_dot_u * x[1] + x[3] * dp)    # upper temperature
        
        cst_l = self.params.cp * m_l
        V_l = self.params.total_volume - x[3]
        dT_l =  (1./cst_l) * (q_dot_l - self.params.cp * m_dot_l * x[2] + V_l * dp)  # lower temperature
        dV_u = (1/ (x[0] * self.params.gamma))* ((self.params.gamma - 1)* (q_dot_u) - x[3] * dp)   # upper volume

        return dp, dT_u, dT_l, dV_u

    def compute_source_terms_heat_flux(self,x):
        """
            Compute q_dot_l, q_dot_u
        """
        q_dot_l = -self.params.cp * self.params.combustion_speed * self.params.combustion_heat * self.params.Q * x[2] # power
        q_dot_u = self.params.combustion_speed * self.params.combustion_heat + self.params.cp * self.params.combustion_speed * self.params.outside_temperature
        q_dot_u += self.params.cp * self.params.combustion_speed * self.params.combustion_heat * self.params.Q * x[2]

        return q_dot_l, q_dot_u

    def compute_source_terms_mass_flux(self):
        """
            Compute m_dot_l, m_dot_u 
        """
        m_dot_u = self.params.combustion_speed * self.params.combustion_heat * self.params.Q + self.params.combustion_speed
        m_dot_l = - self.params.combustion_speed * self.params.combustion_heat
        return m_dot_l, m_dot_u


    # ---------- Plot ----------
    def log_trajectory_plot(self, t_true,y_true, y_pred):
        fig, ax = plt.subplots()

        for i,value in enumerate(["p", "T_u", "T_l", "V_u"]):
            ax.plot(t_true, y_true[:,i], label=f"{value} (SciPy)")
            ax.plot(t_true, y_pred[:, i], '--', label=f"{value} (NN)")

        ax.legend()
        ax.set_xlabel("t")
        ax.set_ylabel("Values")

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        img = torch.tensor(np.array(plt.imread(buf))).permute(2, 0, 1)

        plt.close(fig)
        return img
       

    # ---------- Simple visualization ----------
    def log_trajectory_phase_space_plot(self,y_true, y_pred):

        return None