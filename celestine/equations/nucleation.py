import numpy as np
from numpy.typing import NDArray

from ..state import StateBCs, EQMStateBCs, DISEQStateBCs


def calculate_nucleation(state_BCs: StateBCs, cfg) -> NDArray:
    """implement nucleation term"""
    zeros = np.zeros_like(state_BCs.enthalpy[1:-1])
    match state_BCs:
        case EQMStateBCs():
            return np.hstack((zeros, zeros, zeros))
        case DISEQStateBCs():
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
        case _:
            raise NotImplementedError
