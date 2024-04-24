import numpy as np
import celestine.boundary_conditions as bc
from ..flux import calculate_gas_flux, calculate_heat_flux, calculate_salt_flux
from ..brine_channel_sink_terms import (
    calculate_heat_sink,
    calculate_salt_sink,
    calculate_gas_sink,
)
from .equilibrium_state import EQMState
from .abstract_state_bcs import StateBCs
from ..velocities import calculate_velocities


def prevent_gas_rise_into_saturated_cell(Vg, state_BCs):
    """Modify the gas interstitial velocity to prevent bubble rise into a cell which
    is already theoretically saturated with gas.

    From the state with boundary conditions calculate the gas and solid fraction in the
    cells (except at lower ghost cell). If any of these are such that there is more gas
    fraction than pore space available then set gas insterstitial velocity to zero on
    the edge below. Make sure the very top boundary velocity is not changed as we want
    to always alow flux to the atmosphere regardless of the boundary conditions imposed.

    :param Vg: gas insterstitial velocity on cell edges
    :type Vg: Numpy array (size I+1)
    :param state_BCs: state of system with boundary conditions
    :type state_BCs: celestine.state.StateBCs
    :return: filtered gas interstitial velocities on edges to prevent gas rise into a
        fully gas saturated cell

    """
    gas_fraction_above = state_BCs.gas_fraction[1:]
    solid_fraction_above = 1 - state_BCs.liquid_fraction[1:]
    filtered_Vg = np.where(gas_fraction_above + solid_fraction_above >= 1, 0, Vg)
    filtered_Vg[-1] = Vg[-1]
    return filtered_Vg


class EQMStateBCs(StateBCs):
    """Stores information needed for solution at one timestep with BCs on ghost
    cells as well

    Note must initialise once enthalpy method has already run on State."""

    def __init__(self, state: EQMState):
        """Initialiase the prime variables for the solver:
        enthalpy, bulk salinity and bulk air
        """
        self.cfg = state.cfg
        self.time = state.time
        self.enthalpy = bc.enthalpy_BCs(state.enthalpy, state.cfg)
        self.salt = bc.salt_BCs(state.salt, state.cfg)
        self.gas = bc.gas_BCs(state.gas, state.cfg)

        # here we apply boundary conditions to the secondary variables calculated from
        # the enthalpy method
        self.temperature = bc.temperature_BCs(state.temperature, state.time, state.cfg)
        self.liquid_salinity = bc.liquid_salinity_BCs(state.liquid_salinity, state.cfg)
        self.dissolved_gas = bc.dissolved_gas_BCs(state.dissolved_gas, state.cfg)
        self.gas_fraction = bc.gas_fraction_BCs(state.gas_fraction, state.cfg)
        self.liquid_fraction = bc.liquid_fraction_BCs(state.liquid_fraction, state.cfg)

    def _calculate_brine_convection_sink(self):
        heat_sink = calculate_heat_sink(self, self.cfg)
        salt_sink = calculate_salt_sink(self, self.cfg)
        gas_sink = calculate_gas_sink(self, self.cfg)
        return np.hstack((heat_sink, salt_sink, gas_sink))

    def _calculate_dz_fluxes(self, Wl, Vg, V, D_g, D_e):
        heat_flux = calculate_heat_flux(self, Wl, V, D_g, self.cfg)
        salt_flux = calculate_salt_flux(self, Wl, V, D_g, self.cfg)
        gas_flux = calculate_gas_flux(self, Wl, V, Vg, D_g, self.cfg)
        dz = lambda flux: np.matmul(D_e, flux)
        return np.hstack((dz(heat_flux), dz(salt_flux), dz(gas_flux)))

    def calculate_equation(self, D_g, D_e):
        Vg, Wl, V = calculate_velocities(self)
        Vg = prevent_gas_rise_into_saturated_cell(Vg, self)

        return (
            -self._calculate_dz_fluxes(Wl, Vg, V, D_g, D_e)
            - self._calculate_brine_convection_sink()
        )
