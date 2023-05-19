import numpy as np
from abc import ABC, abstractmethod
from celestine.params import PhysicalParams
from celestine.phase_boundaries import FullPhaseBoundaries, ReducedPhaseBoundaries


class EnthalpyMethod(ABC):
    """Template for an enthalpy method. To implement a new method overwrite the
    initializer to initialise the physical parameters and a suitable phase boundaries
    object. Then implement a calculate enthalpy method that takes a state and uses bulk
    enthalpy, salt and gas to return (temperature, liquid_fraction, gas_fraction,
    solid_fraction, liquid_salinity, dissolved_gas).
    """

    @abstractmethod
    def __init__(self, physical_params: PhysicalParams):
        pass

    @abstractmethod
    def calculate_enthalpy_method(self, state):
        pass


class FullEnthalpyMethod(EnthalpyMethod):
    def __init__(self, physical_params: PhysicalParams):
        """initialise with physical parameters and the full phase boundaries calculator"""
        self.physical_params = physical_params
        self.phase_boundaries = FullPhaseBoundaries(physical_params)

    def calculate_temperature(self, enthalpy, salt, gas, phase_masks):
        chi = self.physical_params.expansion_coefficient
        St = self.physical_params.stefan_number
        C = self.physical_params.concentration_ratio
        temperature = np.full_like(enthalpy, np.NaN)
        l, L, m, M, e, E, s, S = phase_masks
        temperature[l] = enthalpy[l]
        temperature[L] = enthalpy[L] / (1 - ((gas[L] - chi) / (1 - chi)))

        coeff1 = enthalpy[m] + C + St
        coeff2 = C * enthalpy[m] - St * salt[m]
        temperature[m] = (1 / 2) * (coeff1 - np.sqrt(coeff1**2 - 4 * coeff2))

        coeff3 = 1 - gas[M]
        coeff4 = (C + St) * (1 - gas[M]) + enthalpy[M] + salt[M] * chi + C * chi
        coeff5 = C * enthalpy[M] - (1 - chi) * St * salt[M] - C * St * (gas[M] - chi)
        temperature[M] = (1 / (2 * coeff3)) * (
            coeff4 - np.sqrt(coeff4**2 - 4 * coeff3 * coeff5)
        )

        temperature[e] = -1
        temperature[E] = -1

        temperature[s] = enthalpy[s] + St
        temperature[S] = enthalpy[S] / (1 - gas[S]) + St

        return temperature

    def calculate_liquid_fraction(self, enthalpy, salt, gas, temperature, phase_masks):
        chi = self.physical_params.expansion_coefficient
        St = self.physical_params.stefan_number
        C = self.physical_params.concentration_ratio
        liquid_fraction = np.full_like(enthalpy, np.NaN)
        l, L, m, M, e, E, s, S = phase_masks

        liquid_fraction[l] = 1
        liquid_fraction[L] = 1 - ((gas[L] - chi) / (1 - chi))

        liquid_fraction[m] = 1 - (temperature[m] - enthalpy[m]) / St
        liquid_fraction[M] = (salt[M] + C) / (C - temperature[M])

        liquid_fraction[e] = (enthalpy[e] + 1) / St + 1
        liquid_fraction[E] = ((1 - gas[E]) * (1 + St) + enthalpy[E]) / (
            St * (1 - chi) - chi
        )

        liquid_fraction[s] = 0
        liquid_fraction[S] = 0

        return liquid_fraction

    def calculate_gas_fraction(self, gas, liquid_fraction, phase_masks):
        chi = self.physical_params.expansion_coefficient
        gas_fraction = np.full_like(gas, np.NaN)
        l, L, m, M, e, E, s, S = phase_masks

        gas_fraction[l] = 0
        gas_fraction[L] = (gas[L] - chi) / (1 - chi)

        gas_fraction[m] = 0
        gas_fraction[M] = gas[M] - chi * liquid_fraction[M]

        gas_fraction[e] = 0
        gas_fraction[E] = gas[E] - chi * liquid_fraction[E]

        gas_fraction[s] = 0
        gas_fraction[S] = gas[S]

        return gas_fraction

    def calculate_solid_fraction(self, liquid_fraction, gas_fraction):
        solid_fraction = 1 - liquid_fraction - gas_fraction
        return solid_fraction

    def calculate_dissolved_gas(self, gas, liquid_fraction, phase_masks):
        chi = self.physical_params.expansion_coefficient
        dissolved_gas = np.full_like(gas, np.NaN)
        l, L, m, M, e, E, s, S = phase_masks

        dissolved_gas[l] = gas[l] / chi
        dissolved_gas[L] = 1

        dissolved_gas[m] = gas[m] / (chi * liquid_fraction[m])
        dissolved_gas[M] = 1

        dissolved_gas[e] = gas[e] / (chi * liquid_fraction[e])
        dissolved_gas[E] = 1

        dissolved_gas[s] = 1
        dissolved_gas[S] = 1

        return dissolved_gas

    def calculate_liquid_salinity(self, salt, gas, temperature, phase_masks):
        chi = self.physical_params.expansion_coefficient
        C = self.physical_params.concentration_ratio
        liquid_salinity = np.full_like(salt, np.NaN)
        l, L, m, M, e, E, s, S = phase_masks

        liquid_salinity[l] = salt[l]
        gas_fraction = (gas[L] - chi) / (1 - chi)
        liquid_salinity[L] = (salt[L] + gas_fraction * C) / (1 - gas_fraction)

        liquid_salinity[m] = -temperature[m]
        liquid_salinity[M] = -temperature[M]

        liquid_salinity[e] = 1
        liquid_salinity[E] = 1

        liquid_salinity[s] = 1
        liquid_salinity[S] = 1

        return liquid_salinity

    def calculate_enthalpy_method(self, state):
        phase_masks = self.phase_boundaries.get_phase_masks(state)
        enthalpy, salt, gas = state.enthalpy, state.salt, state.gas
        temperature = self.calculate_temperature(enthalpy, salt, gas, phase_masks)
        liquid_fraction = self.calculate_liquid_fraction(
            enthalpy, salt, gas, temperature, phase_masks
        )
        gas_fraction = self.calculate_gas_fraction(gas, liquid_fraction, phase_masks)
        solid_fraction = self.calculate_solid_fraction(liquid_fraction, gas_fraction)
        liquid_salinity = self.calculate_liquid_salinity(
            salt, gas, temperature, phase_masks
        )
        dissolved_gas = self.calculate_dissolved_gas(gas, liquid_fraction, phase_masks)
        return (
            temperature,
            liquid_fraction,
            gas_fraction,
            solid_fraction,
            liquid_salinity,
            dissolved_gas,
        )


class ReducedEnthalpyMethod(EnthalpyMethod):
    def __init__(self, physical_params: PhysicalParams):
        """initialise with physical parameters and the full phase boundaries calculator"""
        self.physical_params = physical_params
        self.phase_boundaries = ReducedPhaseBoundaries(physical_params)

    def calculate_solid_fraction(self, enthalpy, salt, phase_masks):
        solid_fraction = np.full_like(enthalpy, np.NaN)
        L, M, E, S = phase_masks
        St = self.physical_params.stefan_number
        conc = self.physical_params.concentration_ratio

        A = St
        B = enthalpy[M] - St - conc
        C = -(enthalpy[M] + salt[M])

        solid_fraction[L] = 0
        solid_fraction[M] = (1 / (2 * A)) * (-B - np.sqrt(B**2 - 4 * A * C))
        solid_fraction[E] = -(1 + enthalpy[E]) / St
        solid_fraction[S] = 1

        return solid_fraction

    def calculate_temperature(self, enthalpy, solid_fraction, phase_masks):
        L, M, E, S = phase_masks
        St = self.physical_params.stefan_number

        temperature = np.full_like(enthalpy, np.NaN)
        temperature[L] = enthalpy[L]
        temperature[M] = enthalpy[M] + solid_fraction[M] * St
        temperature[E] = -1
        temperature[S] = enthalpy[S] + St

        return temperature

    def calculate_liquid_fraction(self, solid_fraction):
        return 1 - solid_fraction

    def calculate_gas_fraction(self, gas, liquid_fraction):
        chi = self.physical_params.expansion_coefficient
        gas_fraction = np.full_like(liquid_fraction, np.NaN)

        gas_sat = chi * liquid_fraction
        gas_fraction = np.where(gas >= gas_sat, gas - gas_sat, 0)
        return gas_fraction

    def calculate_dissolved_gas(self, gas, liquid_fraction):
        chi = self.physical_params.expansion_coefficient
        dissolved_gas = np.full_like(gas, np.NaN)

        gas_sat = chi * liquid_fraction
        dissolved_gas = np.where(gas >= gas_sat, 1, gas / gas_sat)
        return dissolved_gas

    def calculate_liquid_salinity(self, salt, temperature, phase_masks):
        liquid_salinity = np.full_like(salt, np.NaN)
        L, M, E, S = phase_masks

        liquid_salinity[L] = salt[L]
        liquid_salinity[M] = -temperature[M]
        liquid_salinity[E] = 1
        liquid_salinity[S] = 1

        return liquid_salinity

    def calculate_enthalpy_method(self, state):
        phase_masks = self.phase_boundaries.get_phase_masks(state)
        enthalpy, salt, gas = state.enthalpy, state.salt, state.gas
        solid_fraction = self.calculate_solid_fraction(enthalpy, salt, phase_masks)
        liquid_fraction = self.calculate_liquid_fraction(solid_fraction)
        temperature = self.calculate_temperature(enthalpy, solid_fraction, phase_masks)
        gas_fraction = self.calculate_gas_fraction(gas, liquid_fraction)
        liquid_salinity = self.calculate_liquid_salinity(salt, temperature, phase_masks)
        dissolved_gas = self.calculate_dissolved_gas(gas, liquid_fraction)
        return (
            temperature,
            liquid_fraction,
            gas_fraction,
            solid_fraction,
            liquid_salinity,
            dissolved_gas,
        )


def get_enthalpy_method(cfg):
    solver_choice = cfg.numerical_params.solver
    options = {"LU": FullEnthalpyMethod, "RED": ReducedEnthalpyMethod}
    return options[solver_choice]
