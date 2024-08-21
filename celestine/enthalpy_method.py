"""Module containing enthalpy method to calculate state variables from bulk enthalpy,
bulk salinity and bulk gas."""

import numpy as np
from celestine.params import PhysicalParams, Config
from celestine.phase_boundaries import get_phase_masks
from .state import EQMState, DISEQState, State, StateFull, EQMStateFull, DISEQStateFull


def calculate_enthalpy_method(cfg: Config, state: State) -> StateFull:
    physical_params = cfg.physical_params
    phase_masks = get_phase_masks(state, physical_params)
    solid_fraction = _calculate_solid_fraction(state, physical_params, phase_masks)
    liquid_fraction = _calculate_liquid_fraction(solid_fraction)
    temperature = _calculate_temperature(
        state, solid_fraction, physical_params, phase_masks
    )
    liquid_salinity = _calculate_liquid_salinity(state, temperature, phase_masks)
    dissolved_gas = _calculate_dissolved_gas(
        state, liquid_fraction, physical_params, phase_masks
    )
    gas_fraction = _calculate_gas_fraction(state, liquid_fraction, physical_params)

    match state:
        case EQMState():
            return EQMStateFull(
                state.time,
                state.enthalpy,
                state.salt,
                state.gas,
                temperature,
                liquid_fraction,
                solid_fraction,
                liquid_salinity,
                dissolved_gas,
                gas_fraction,
            )
        case DISEQState():
            return DISEQStateFull(
                state.time,
                state.enthalpy,
                state.salt,
                state.bulk_dissolved_gas,
                state.gas_fraction,
                temperature,
                liquid_fraction,
                solid_fraction,
                liquid_salinity,
                dissolved_gas,
            )
        case _:
            raise NotImplementedError


def _calculate_solid_fraction(state, physical_params: PhysicalParams, phase_masks):
    enthalpy, salt = state.enthalpy, state.salt
    solid_fraction = np.full_like(enthalpy, np.NaN)
    L, M, E, S = phase_masks
    St = physical_params.stefan_number
    conc = physical_params.concentration_ratio

    A = St
    B = enthalpy[M] - St - conc
    C = -(enthalpy[M] + salt[M])

    solid_fraction[L] = 0
    solid_fraction[M] = (1 / (2 * A)) * (-B - np.sqrt(B**2 - 4 * A * C))
    solid_fraction[E] = -(1 + enthalpy[E]) / St
    solid_fraction[S] = 1

    return solid_fraction


def _calculate_temperature(
    state, solid_fraction, physical_params: PhysicalParams, phase_masks
):
    enthalpy = state.enthalpy
    L, M, E, S = phase_masks
    St = physical_params.stefan_number

    temperature = np.full_like(enthalpy, np.NaN)
    temperature[L] = enthalpy[L]
    temperature[M] = enthalpy[M] + solid_fraction[M] * St
    temperature[E] = -1
    temperature[S] = enthalpy[S] + St

    return temperature


def _calculate_liquid_fraction(solid_fraction):
    return 1 - solid_fraction


def _calculate_liquid_salinity(state, temperature, phase_masks):
    salt = state.salt
    liquid_salinity = np.full_like(salt, np.NaN)
    L, M, E, S = phase_masks

    liquid_salinity[L] = salt[L]
    liquid_salinity[M] = -temperature[M]
    liquid_salinity[E] = 1
    liquid_salinity[S] = 1

    return liquid_salinity


def _calculate_gas_fraction(state, liquid_fraction, physical_params: PhysicalParams):
    match state:
        case EQMState():
            chi = physical_params.expansion_coefficient
            tolerable_super_saturation = (
                physical_params.tolerable_super_saturation_fraction
            )
            gas_fraction = np.full_like(liquid_fraction, np.NaN)

            gas_sat = chi * liquid_fraction * tolerable_super_saturation
            is_super_saturated = state.gas >= gas_sat
            is_sub_saturated = ~is_super_saturated
            gas_fraction[is_super_saturated] = (
                state.gas[is_super_saturated] - gas_sat[is_super_saturated]
            )
            gas_fraction[is_sub_saturated] = 0
            return gas_fraction
        case DISEQState():
            return state.gas_fraction
        case _:
            raise NotImplementedError


def _calculate_dissolved_gas(
    state, liquid_fraction, physical_params: PhysicalParams, phase_masks
):
    chi = physical_params.expansion_coefficient
    match state:
        case EQMState():
            gas = state.gas
            tolerable_super_saturation = (
                physical_params.tolerable_super_saturation_fraction
            )
            dissolved_gas = np.full_like(gas, np.NaN)

            gas_sat = chi * liquid_fraction * tolerable_super_saturation
            is_super_saturated = gas >= gas_sat
            is_sub_saturated = ~is_super_saturated
            dissolved_gas[is_super_saturated] = tolerable_super_saturation
            dissolved_gas[is_sub_saturated] = gas[is_sub_saturated] / (
                chi * liquid_fraction[is_sub_saturated]
            )
            return dissolved_gas
        case DISEQState():
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
        case _:
            raise NotImplementedError
