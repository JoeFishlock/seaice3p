from typing import Callable
import numpy as np
from numpy.typing import NDArray

from ..state import StateBCs, EQMStateBCs, DISEQStateBCs
from ..params import Config, EQMPhysicalParams, DISEQPhysicalParams


def get_nucleation(cfg: Config) -> Callable[[StateBCs], NDArray]:
    fun_map = {
        EQMPhysicalParams: _EQM_nucleation,
        DISEQPhysicalParams: _DISEQ_nucleation,
    }

    def nucleation(state_BCs: StateBCs) -> NDArray:
        return fun_map[type(cfg.physical_params)](state_BCs, cfg)

    return nucleation


def _EQM_nucleation(state_BCs: EQMStateBCs, cfg: Config) -> NDArray:
    """implement nucleation term"""
    zeros = np.zeros_like(state_BCs.enthalpy[1:-1])
    return np.hstack((zeros, zeros, zeros))


def _DISEQ_nucleation(state_BCs: DISEQStateBCs, cfg: Config) -> NDArray:
    """implement nucleation term"""
    zeros = np.zeros_like(state_BCs.enthalpy[1:-1])
    chi = cfg.physical_params.expansion_coefficient
    Da = cfg.physical_params.damkohler_number
    centers = np.s_[1:-1]
    bulk_dissolved_gas = state_BCs.bulk_dissolved_gas[centers]
    liquid_fraction = state_BCs.liquid_fraction[centers]
    saturation = chi * liquid_fraction
    gas_fraction = state_BCs.gas_fraction[centers]

    is_saturated = bulk_dissolved_gas > saturation
    nucleation = np.full_like(bulk_dissolved_gas, np.NaN)
    nucleation[is_saturated] = Da * (
        bulk_dissolved_gas[is_saturated] - saturation[is_saturated]
    )
    nucleation[~is_saturated] = -Da * gas_fraction[~is_saturated]

    return np.hstack(
        (
            zeros,
            zeros,
            -nucleation,
            nucleation,
        )
    )
