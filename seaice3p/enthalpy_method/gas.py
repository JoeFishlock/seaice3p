import numpy as np
from numpy.typing import NDArray
from ..params import PhysicalParams
from ..state import EQMState, DISEQState


def calculate_EQM_gas_fraction(
    state: EQMState, liquid_fraction: NDArray, physical_params: PhysicalParams
) -> NDArray:
    chi = physical_params.expansion_coefficient
    tolerable_super_saturation = physical_params.tolerable_super_saturation_fraction
    gas_fraction = np.full_like(liquid_fraction, np.NaN)

    gas_sat = chi * liquid_fraction * tolerable_super_saturation
    is_super_saturated = state.gas >= gas_sat
    is_sub_saturated = ~is_super_saturated
    gas_fraction[is_super_saturated] = (
        state.gas[is_super_saturated] - gas_sat[is_super_saturated]
    )
    gas_fraction[is_sub_saturated] = 0
    return gas_fraction


def calculate_EQM_dissolved_gas(
    state: EQMState, liquid_fraction, physical_params: PhysicalParams
) -> NDArray:
    chi = physical_params.expansion_coefficient
    gas = state.gas
    tolerable_super_saturation = physical_params.tolerable_super_saturation_fraction
    dissolved_gas = np.zeros_like(gas)

    # If no dissolved phase
    if chi == 0:
        return dissolved_gas

    gas_sat = chi * liquid_fraction * tolerable_super_saturation
    is_super_saturated = gas >= gas_sat
    is_sub_saturated = ~is_super_saturated
    dissolved_gas[is_super_saturated] = tolerable_super_saturation
    dissolved_gas[is_sub_saturated] = gas[is_sub_saturated] / (
        chi * liquid_fraction[is_sub_saturated]
    )
    return dissolved_gas


def calculate_DISEQ_dissolved_gas(
    state: DISEQState, liquid_fraction, physical_params: PhysicalParams, phase_masks
) -> NDArray:
    chi = physical_params.expansion_coefficient
    L, M, E, S = phase_masks
    bulk_dissolved_gas = state.bulk_dissolved_gas

    # prevent dissolved gas concentration blowing up during total freezing
    REGULARISATION = 1e-6

    dissolved_gas = np.full_like(bulk_dissolved_gas, np.NaN)
    dissolved_gas[L] = bulk_dissolved_gas[L] / chi
    dissolved_gas[M] = bulk_dissolved_gas[M] / (chi * liquid_fraction[M])
    dissolved_gas[E] = bulk_dissolved_gas[E] / (
        chi * liquid_fraction[E] + REGULARISATION
    )
    dissolved_gas[S] = 0
    return dissolved_gas
