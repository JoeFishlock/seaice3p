"""Module containing enthalpy method to calculate state variables from bulk enthalpy,
bulk salinity and bulk gas."""

import numpy as np
from abc import ABC, abstractmethod
from celestine.params import PhysicalParams, Config
from celestine.phase_boundaries import ReducedPhaseBoundaries
from celestine.state.disequilibrium_state import DISEQStateFull
from .state import EQMState, DISEQState, State, StateFull, EQMStateFull


def calculate_enthalpy_method(cfg: Config, state: State) -> StateFull:
    match state:
        case EQMState():
            (
                temperature,
                liquid_fraction,
                gas_fraction,
                solid_fraction,
                liquid_salinity,
                dissolved_gas,
            ) = ReducedEnthalpyMethod(cfg.physical_params).calculate_enthalpy_method(
                state
            )
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
            (
                temperature,
                liquid_fraction,
                gas_fraction,
                solid_fraction,
                liquid_salinity,
                dissolved_gas,
            ) = DisequilibriumEnthalpyMethod(
                cfg.physical_params
            ).calculate_enthalpy_method(
                state
            )
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
        tolerable_super_saturation = (
            self.physical_params.tolerable_super_saturation_fraction
        )
        gas_fraction = np.full_like(liquid_fraction, np.NaN)

        gas_sat = chi * liquid_fraction * tolerable_super_saturation
        is_super_saturated = gas >= gas_sat
        is_sub_saturated = ~is_super_saturated
        gas_fraction[is_super_saturated] = (
            gas[is_super_saturated] - gas_sat[is_super_saturated]
        )
        gas_fraction[is_sub_saturated] = 0
        return gas_fraction

    def calculate_dissolved_gas(self, gas, liquid_fraction):
        chi = self.physical_params.expansion_coefficient
        tolerable_super_saturation = (
            self.physical_params.tolerable_super_saturation_fraction
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


class DisequilibriumEnthalpyMethod(ReducedEnthalpyMethod):
    """IGNORE TOLERABLE SUPERSATURATION PARAMETER FOR THIS IMPLEMENTATION"""

    def calculate_gas_fraction(self):
        raise TypeError("No need to call this as gas fraction is a primary variable")

    def calculate_dissolved_gas(self, bulk_dissolved_gas, liquid_fraction, phase_masks):
        chi = self.physical_params.expansion_coefficient
        L, M, E, S = phase_masks

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

    def calculate_enthalpy_method(self, state):
        phase_masks = self.phase_boundaries.get_phase_masks(state)
        enthalpy, salt, bulk_dissolved_gas, gas_fraction = (
            state.enthalpy,
            state.salt,
            state.bulk_dissolved_gas,
            state.gas_fraction,
        )
        solid_fraction = self.calculate_solid_fraction(enthalpy, salt, phase_masks)
        liquid_fraction = self.calculate_liquid_fraction(solid_fraction)
        temperature = self.calculate_temperature(enthalpy, solid_fraction, phase_masks)
        liquid_salinity = self.calculate_liquid_salinity(salt, temperature, phase_masks)
        dissolved_gas = self.calculate_dissolved_gas(
            bulk_dissolved_gas, liquid_fraction, phase_masks
        )
        return (
            temperature,
            liquid_fraction,
            gas_fraction,
            solid_fraction,
            liquid_salinity,
            dissolved_gas,
        )
