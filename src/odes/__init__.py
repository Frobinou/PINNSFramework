from src.odes.ode_repository.ode_lotka_voltera import LotkaVoltera
from src.odes.ode_repository.ode_cfast import ODECFAST
from src.data_models import AvailablesODE

ODE_REPOSITORY = {
    AvailablesODE.LOTKA_VOLTERA : LotkaVoltera,
    AvailablesODE.CFAST : ODECFAST
}