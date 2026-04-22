import torch.nn as nn
from src.repositories.models import AvailablesAIModel
from src.core.registry import REGISTRY  

@REGISTRY.models.register(AvailablesAIModel.BASIC_PINN)
class BasicPINN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=20, output_dim=1):
        self.name = AvailablesAIModel.BASIC_PINN
        super(BasicPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
