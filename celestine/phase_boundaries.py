"""Module for calculating the phase boundaries needed for the enthalpy method.
"""

import numpy as np
from abc import ABC, abstractmethod
from celestine.params import PhysicalParams


class PhaseBoundaries(ABC):
    """Template for phase boundary calculation.

    Concrete implementations should use the state containing enthalpy, salt and gas to
    calculate the liquidus, enthalpy, solidus and saturation boundaries and then return
    masks for each possible phase of the system.
    """

    def __init__(self, physical_params: PhysicalParams):
        self.physical_params = physical_params

    @abstractmethod
    def get_phase_masks(self, state):
        pass


class ReducedPhaseBoundaries(PhaseBoundaries):
    r"""calculates the phase boundaries neglecting the gas fraction so that

    .. math:: \phi_s + \phi_l = 1

    """

    def calculate_liquidus(self, salt):
        return -salt

    def calculate_eutectic(self, salt):
        C = self.physical_params.concentration_ratio
        St = self.physical_params.stefan_number
        return (St * (salt - 1) / (1 + C)) - 1

    def calculate_solidus(self, salt):
        St = self.physical_params.stefan_number
        return np.full_like(salt, -1 - St)

    def get_phase_masks(self, state):
        enthalpy, salt = state.enthalpy, state.salt
        liquidus = self.calculate_liquidus(salt)
        eutectic = self.calculate_eutectic(salt)
        solidus = self.calculate_solidus(salt)
        is_liquid = enthalpy >= liquidus
        is_mush = (enthalpy >= eutectic) & (enthalpy < liquidus)
        is_eutectic = (enthalpy >= solidus) & (enthalpy < eutectic)
        is_solid = enthalpy < solidus
        L = is_liquid
        M = is_mush
        E = is_eutectic
        S = is_solid
        return L, M, E, S
