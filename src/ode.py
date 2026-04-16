from dataclasses import dataclass 
import torch 
from scipy.integrate import solve_ivp
import numpy as np 

@dataclass
class Params:
    alpha: float = 1.
    beta: float = 2.

def ode_lotka_voltera(y: torch.Tensor,params: Params) -> torch.Tensor:
    # A generic example of ode to be solved - here a Lotka-Voltera
    n = y.shape[1]
    dy = torch.zeros_like(y)
    dy[:,0] = params.alpha  *  x_pred - params.beta* x_pred * y_predator
    dy[:,1] = params.delta * x_pred * y_predator - params.gamma * y_predator
    return dy


def validation_lotka_voltera(t_span = (0, 20),z0= [10, 5], params:Params):
    t_eval = np.linspace(*t_span, 200)
    sol = solve_ivp(lambda y:lotka_volterra(t,z, params), t_span, z0, t_eval=t_eval)
    return sol 

def lotka_volterra(t, z, params):
    x, y = z
    dxdt = params.alpha*x - params.beta*x*y
    dydt = params.delta*x*y - params.gamma*y
    return [dxdt, dydt]