r"""Module for calculating the phase boundaries needed for the enthalpy method.

calculates the phase boundaries neglecting the gas fraction so that

    .. math:: \phi_s + \phi_l = 1

"""

import numpy as np
from ..state import State
from ..params import PhysicalParams


def _calculate_liquidus(salt):
    return -salt


def _calculate_eutectic(salt, physical_params: PhysicalParams):
    C = physical_params.concentration_ratio
    St = physical_params.stefan_number
    ratio = physical_params.specific_heat_ratio
    return ((St + ratio - 1) * (salt - 1) / (1 + C)) - 1


def _calculate_solidus(salt, physical_params: PhysicalParams):
    St = physical_params.stefan_number
    ratio = physical_params.specific_heat_ratio
    return np.full_like(salt, -ratio - St)


def get_phase_masks(state: State, physical_params: PhysicalParams):
    enthalpy, salt = state.enthalpy, state.salt
    concentration_ratio, stefan_number = (
        physical_params.concentration_ratio,
        physical_params.stefan_number,
    )
    liquidus = _calculate_liquidus(salt)
    eutectic = _calculate_eutectic(salt, physical_params)
    solidus = _calculate_solidus(salt, physical_params)
    is_liquid = enthalpy >= liquidus
    is_mush = (enthalpy >= eutectic) & (enthalpy < liquidus)
    is_eutectic = (enthalpy >= solidus) & (enthalpy < eutectic)
    is_solid = enthalpy < solidus
    L = is_liquid
    M = is_mush
    E = is_eutectic
    S = is_solid
    return L, M, E, S
