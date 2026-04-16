import sys
from pathlib import Path
cwd = str(Path.cwd())
sys.path.append(cwd)

from src.trainer_runner import Trainer
from src.data_models import ParametersTraining, ODESpecifications, AvailablesODE, AvailablesAIModel

case = AvailablesODE.CFAST


if case == AvailablesODE.LOTKA_VOLTERA:
    from src.ode_repository.ode_lotka_voltera import ParamsLotkaVoltera
    trainer = Trainer(
        parameters_training=ParametersTraining(l_r=1e-3, epochs=200),
        ode_specifications=ODESpecifications(ode_name=AvailablesODE.LOTKA_VOLTERA,
                                            initial_conditions=[10.,10.],
                                            grid_size=800,
                                            model_dimension=2),
        ode_parameters= ParamsLotkaVoltera(alpha=2/3, beta=4/3, gamma=1, delta=1),
        output_folder_path= Path('runs') / f'{AvailablesODE.LOTKA_VOLTERA}', 
        model_name= AvailablesAIModel.BASIC_PINN
    )
else:
    from src.ode_repository.ode_cfast import ParamsCFAST
    trainer = Trainer(
        parameters_training=ParametersTraining(l_r=1e-3, epochs=4000),
        ode_specifications=ODESpecifications(ode_name=AvailablesODE.CFAST,
                                            initial_conditions=[101325.,293., 293., 0.], # [p, T_u, T_l, V_u]
                                            grid_size=800,
                                            model_dimension=4),
        ode_parameters= ParamsCFAST(total_volume=250),
        output_folder_path= Path('runs') / f'{AvailablesODE.CFAST}', 
        model_name= AvailablesAIModel.BASIC_PINN
    )
trainer.run()