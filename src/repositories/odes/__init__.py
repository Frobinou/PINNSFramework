try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class AvailablesODE(StrEnum):
    LOTKA_VOLTERA = "Lotka-Voltera"
    CFAST = "CFAST"