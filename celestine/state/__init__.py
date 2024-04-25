from .equilibrium_state import EQMState
from .disequilibrium_state import DISEQState


def get_model(cfg):
    MODEL_CHOICES = {"EQM": EQMState, "DISEQ": DISEQState}
    return MODEL_CHOICES[cfg.model]
