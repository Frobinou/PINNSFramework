import sys
from pathlib import Path
cwd = str(Path.cwd())
sys.path.append(cwd)

from src.trainer import Trainer
from pathlib import Path
from src.data_models import ParametersTraining, ODESpecifications, AvailablesODE, AvailablesAIModel


from src.ode_repository.ode_lotka_voltera import ParamsLotkaVoltera
trainer = Trainer(
    parameters_training=ParametersTraining(l_r=1e-3, epochs=4000),
    ode_specifications=ODESpecifications(ode_name=AvailablesODE.LOTKA_VOLTERA,
                                         initial_conditions=[10.,10.],
                                         grid_size=800,
                                         model_dimension=2),
    ode_parameters= ParamsLotkaVoltera(alpha=2/3, beta=4/3, gamma=1, delta=1),
    output_folder_path= Path('runs') / f'{AvailablesODE.LOTKA_VOLTERA}', 
    model_name= AvailablesAIModel.BASIC_PINN
)

trainer.run()