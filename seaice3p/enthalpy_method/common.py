from typing import Tuple
from numpy.typing import NDArray
import numpy as np
from ..state import State
from ..params import Config, PhysicalParams
from .phase_boundaries import get_phase_masks


def calculate_common_enthalpy_method_vars(
    state: State, cfg: Config, phase_masks
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    physical_params = cfg.physical_params
    phase_masks = get_phase_masks(state, physical_params)
    solid_fraction = _calculate_solid_fraction(state, physical_params, phase_masks)
    liquid_fraction = _calculate_liquid_fraction(solid_fraction)
    temperature = _calculate_temperature(
        state, solid_fraction, physical_params, phase_masks
    )
    liquid_salinity = _calculate_liquid_salinity(state, temperature, phase_masks)
    return solid_fraction, liquid_fraction, temperature, liquid_salinity


def _calculate_solid_fraction(state, physical_params: PhysicalParams, phase_masks):
    enthalpy, salt = state.enthalpy, state.salt
    solid_fraction = np.full_like(enthalpy, np.NaN)
    L, M, E, S = phase_masks
    St = physical_params.stefan_number
    conc = physical_params.concentration_ratio
    ratio = physical_params.specific_heat_ratio

    A = St + conc * (1 - ratio)
    B = enthalpy[M] - St - conc + salt[M] * (1 - ratio)
    C = -(enthalpy[M] + salt[M])

    solid_fraction[L] = 0
    solid_fraction[M] = (1 / (2 * A)) * (-B - np.sqrt(B**2 - 4 * A * C))
    solid_fraction[E] = -(1 + enthalpy[E]) / (St + ratio - 1)
    solid_fraction[S] = 1

    return solid_fraction


def _calculate_temperature(
    state, solid_fraction, physical_params: PhysicalParams, phase_masks
):
    enthalpy = state.enthalpy
    L, M, E, S = phase_masks
    St = physical_params.stefan_number
    ratio = physical_params.specific_heat_ratio

    temperature = np.full_like(enthalpy, np.NaN)
    temperature[L] = enthalpy[L]
    temperature[M] = (enthalpy[M] + solid_fraction[M] * St) / (
        1 + (ratio - 1) * solid_fraction[M]
    )
    temperature[E] = -1
    temperature[S] = (enthalpy[S] + St) / ratio

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
