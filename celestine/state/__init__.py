from typing import Callable
import numpy as np
from numpy.typing import NDArray
from .equilibrium_state import EQMState, EQMStateFull, EQMStateBCs
from .disequilibrium_state import DISEQState, DISEQStateFull, DISEQStateBCs

State = EQMState | DISEQState
StateFull = EQMStateFull | DISEQStateFull
StateBCs = EQMStateBCs | DISEQStateBCs


def get_unpacker(cfg) -> Callable[[float, NDArray], State]:
    fun_map = {
        "EQM": _unpack_EQM,
        "DISEQ": _unpack_DISEQ,
    }

    def unpack(time, solution_vector) -> State:
        return fun_map[cfg.model](time, solution_vector)

    return unpack


def _unpack_EQM(time, solution_vector) -> EQMState:
    enthalpy, salt, gas = np.split(solution_vector, 3)
    return EQMState(time, enthalpy, salt, gas)


def _unpack_DISEQ(time, solution_vector) -> DISEQState:
    enthalpy, salt, bulk_dissolved_gas, gas_fraction = np.split(solution_vector, 4)
    return DISEQState(time, enthalpy, salt, bulk_dissolved_gas, gas_fraction)
