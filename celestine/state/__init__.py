import numpy as np
from .equilibrium_state import EQMState, EQMStateFull, EQMStateBCs
from .disequilibrium_state import DISEQState, DISEQStateFull, DISEQStateBCs

State = EQMState | DISEQState
StateFull = EQMStateFull | DISEQStateFull
StateBCs = EQMStateBCs | DISEQStateBCs


def get_model(cfg) -> State:
    MODEL_CHOICES = {"EQM": EQMState, "DISEQ": DISEQState}
    return MODEL_CHOICES[cfg.model]


def get_state(cfg, time, solution_vector) -> State:
    model_choice = cfg.model
    match model_choice:
        case "EQM":
            enthalpy, salt, gas = np.split(solution_vector, 3)
            return EQMState(time, enthalpy, salt, gas)
        case "DISEQ":
            enthalpy, salt, bulk_dissolved_gas, gas_fraction = np.split(
                solution_vector, 4
            )
            return DISEQState(time, enthalpy, salt, bulk_dissolved_gas, gas_fraction)
        case _:
            raise NotImplementedError
