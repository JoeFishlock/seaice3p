import numpy as np
import celestine.boundary_conditions as bc
from ..flux import (
    calculate_heat_flux,
    calculate_salt_flux,
    calculate_bulk_dissolved_gas_flux,
    calculate_gas_fraction_flux,
)
from ..RJW14 import (
    calculate_heat_sink,
    calculate_salt_sink,
    calculate_bulk_dissolved_gas_sink,
)
from .abstract_state_bcs import StateBCs
from .equilibrium_state_bcs import prevent_gas_rise_into_saturated_cell
from ..velocities import calculate_velocities


class DISEQStateBCs(StateBCs):
    """Stores information needed for solution at one timestep with BCs on ghost
    cells as well

    Note must initialise once enthalpy method has already run on State."""

    def __init__(self, state):
        """Initialiase the prime variables for the solver:
        enthalpy, bulk salinity and bulk air
        """
        self.cfg = state.cfg
        self.time = state.time
        self.enthalpy = bc.enthalpy_BCs(state.enthalpy, state.cfg)
        self.salt = bc.salt_BCs(state.salt, state.cfg)

        # here we apply boundary conditions to the secondary variables calculated from
        # the enthalpy method
        self.temperature = bc.temperature_BCs(state.temperature, state.time, state.cfg)
        self.liquid_salinity = bc.liquid_salinity_BCs(state.liquid_salinity, state.cfg)
        self.dissolved_gas = bc.dissolved_gas_BCs(state.dissolved_gas, state.cfg)
        self.liquid_fraction = bc.liquid_fraction_BCs(state.liquid_fraction, state.cfg)

        self.bulk_dissolved_gas = (
            state.cfg.physical_params.expansion_coefficient
            * self.liquid_fraction
            * self.dissolved_gas
        )
        self.gas_fraction = bc.gas_fraction_BCs(state.gas_fraction, state.cfg)

    def _calculate_brine_convection_sink(self):
        """TODO: check the sink terms for bulk_dissolved_gas and gas fraction

        For now neglect the coupling of bubbles to the horizontal or vertical flow
        """
        heat_sink = calculate_heat_sink(self)
        salt_sink = calculate_salt_sink(self)
        bulk_dissolved_gas_sink = calculate_bulk_dissolved_gas_sink(self)
        gas_fraction_sink = np.zeros_like(heat_sink)
        return np.hstack(
            (heat_sink, salt_sink, bulk_dissolved_gas_sink, gas_fraction_sink)
        )

    def _calculate_nucleation(self):
        """implement nucleation term"""
        chi = self.cfg.physical_params.expansion_coefficient
        Da = 1
        centers = np.s_[1:-1]
        bulk_dissolved_gas = self.bulk_dissolved_gas[centers]
        liquid_fraction = self.liquid_fraction[centers]
        saturation = chi * liquid_fraction
        gas_fraction = self.gas_fraction[centers]

        is_saturated = bulk_dissolved_gas > saturation
        nucleation = np.full_like(bulk_dissolved_gas, np.NaN)
        nucleation[is_saturated] = Da * (
            bulk_dissolved_gas[is_saturated] - saturation[is_saturated]
        )
        nucleation[~is_saturated] = -Da * gas_fraction[~is_saturated]

        return np.hstack(
            (
                np.zeros_like(nucleation),
                np.zeros_like(nucleation),
                -nucleation,
                nucleation,
            )
        )

    def _calculate_dz_fluxes(self, Wl, Vg, V, D_g, D_e):
        heat_flux = calculate_heat_flux(self, Wl, V, D_g, self.cfg)
        salt_flux = calculate_salt_flux(self, Wl, V, D_g, self.cfg)
        bulk_dissolved_gas_flux = calculate_bulk_dissolved_gas_flux(
            self, Wl, V, D_g, self.cfg
        )
        gas_fraction_flux = calculate_gas_fraction_flux(self, V, Vg)
        dz = lambda flux: np.matmul(D_e, flux)
        return np.hstack(
            (
                dz(heat_flux),
                dz(salt_flux),
                dz(bulk_dissolved_gas_flux),
                dz(gas_fraction_flux),
            )
        )

    def calculate_equation(self, D_g, D_e):
        Vg, Wl, V = calculate_velocities(self)
        Vg = prevent_gas_rise_into_saturated_cell(Vg, self)

        return (
            -self._calculate_dz_fluxes(Wl, Vg, V, D_g, D_e)
            - self._calculate_brine_convection_sink()
            + self._calculate_nucleation()
        )
