
import enum


class ParamType(enum.Enum):
    """Enum class for parameter types"""
    STRING = 1
    INTEGER = 2
    FLOAT = 3
    BOOLEAN = 4


class Params(enum.Enum):
    """Enum class for parameter names"""
    LEARNING_RATE = 1
    BATCH_SIZE = 2
    ACTIVATION_FUNCTION = 3
    Epochs = 4


class ParamHandler:
    # Needs:
    # - function that returns requested params
    # -
    pass
