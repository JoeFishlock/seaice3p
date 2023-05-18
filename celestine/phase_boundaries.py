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


class FullPhaseBoundaries(PhaseBoundaries):
    """calculates the phase boundaries when we include gas fraction in bulk enthalpy
    and bulk salinity.
    """

    def calculate_liquidus(self, salt, gas):
        liquidus = np.full_like(salt, np.NaN)
        chi = self.physical_params.expansion_coefficient
        C = self.physical_params.concentration_ratio
        is_sub = gas <= chi
        is_super = ~is_sub
        liquidus[is_sub] = -salt[is_sub]
        liquidus[is_super] = -(salt[is_super] + C * ((gas[is_super] - chi) / (1 - chi)))
        return liquidus

    def calculate_eutectic(self, salt, gas):
        eutectic = np.full_like(salt, np.NaN)
        chi = self.physical_params.expansion_coefficient
        C = self.physical_params.concentration_ratio
        St = self.physical_params.stefan_number
        eutectic_liquid_fraction = (salt + C) / (1 + C)
        is_sub = gas <= chi * eutectic_liquid_fraction
        is_super = ~is_sub
        eutectic[is_sub] = -1 - St * (1 - eutectic_liquid_fraction[is_sub])
        eutectic[is_super] = (
            -(1 - gas[is_super] + chi * eutectic_liquid_fraction[is_super])
            - (1 - gas[is_super] + eutectic_liquid_fraction[is_super] * (chi - 1)) * St
        )
        return eutectic

    def calculate_solidus(self, salt, gas):
        solidus = np.full_like(salt, np.NaN)
        St = self.physical_params.stefan_number
        is_sub = gas <= 0
        is_super = ~is_sub
        solidus[is_sub] = -1 - St
        solidus[is_super] = (1 - gas[is_super]) * (-1 - St)
        return solidus

    def calculate_saturation(self, enthalpy, salt):
        chi = self.physical_params.expansion_coefficient
        St = self.physical_params.stefan_number
        C = self.physical_params.concentration_ratio
        saturation = np.full_like(enthalpy, np.NaN)
        no_gas = np.zeros_like(salt)
        liquidus = self.calculate_liquidus(salt, no_gas)
        eutectic = self.calculate_eutectic(salt, no_gas)
        solidus = self.calculate_solidus(salt, no_gas)
        is_liquid = enthalpy >= liquidus
        is_mush = (enthalpy >= eutectic) & (enthalpy < liquidus)
        is_eutectic = (enthalpy >= solidus) & (enthalpy < eutectic)
        is_solid = enthalpy < solidus
        saturation[is_liquid] = chi
        B = enthalpy[is_mush] + C + St
        A = C * enthalpy[is_mush] - St * salt[is_mush]
        mush_temperature = (1 / 2) * (B - np.sqrt(B**2 - 4 * A))
        saturation[is_mush] = chi * (1 - ((mush_temperature - enthalpy[is_mush]) / St))
        saturation[is_eutectic] = chi * (1 + (enthalpy[is_eutectic] + 1) / St)
        saturation[is_solid] = 0
        return saturation

    def calculate_max_salt(self, gas):
        max_salt = np.full_like(gas, np.NaN)
        chi = self.physical_params.expansion_coefficient
        C = self.physical_params.concentration_ratio
        is_sub = gas <= chi
        is_super = ~is_sub
        max_salt[is_sub] = 1
        max_salt[is_super] = 1 - (gas[is_super] - chi) / (1 - chi) * (1 + C)
        return max_salt

    def get_phase_masks(self, state):
        enthalpy, salt, gas = state.enthalpy, state.salt, state.gas
        liquidus = self.calculate_liquidus(salt, gas)
        eutectic = self.calculate_eutectic(salt, gas)
        solidus = self.calculate_solidus(salt, gas)
        saturation = self.calculate_saturation(enthalpy, salt)
        is_liquid = enthalpy >= liquidus
        is_mush = (enthalpy >= eutectic) & (enthalpy < liquidus)
        is_eutectic = (enthalpy >= solidus) & (enthalpy < eutectic)
        is_solid = enthalpy < solidus
        is_sub = gas <= saturation
        is_super = ~is_sub
        l = is_liquid & is_sub
        L = is_liquid & is_super
        m = is_mush & is_sub
        M = is_mush & is_super
        e = is_eutectic & is_sub
        E = is_eutectic & is_super
        s = is_solid & is_sub
        S = is_solid & is_super
        return l, L, m, M, e, E, s, S
