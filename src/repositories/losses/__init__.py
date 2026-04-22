try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass

class AvailablesLoss(StrEnum):
    PINN_LOSS = "PINN_LOSS"