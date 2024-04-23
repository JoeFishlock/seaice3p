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


class EQMStateBCs(StateBCs):
    """Stores information needed for solution at one timestep with BCs on ghost
    cells as well

    Note must initialise once enthalpy method has already run on State."""

    def __init__(self, state: EQMState):
        self.cfg = state.cfg
        self.time = state.time
        self.enthalpy = bc.enthalpy_BCs(state.enthalpy, state.cfg)
        self.salt = bc.salt_BCs(state.salt, state.cfg)
        self.gas = bc.gas_BCs(state.gas, state.cfg)

        if state.pressure is not None:
            self.pressure = bc.pressure_BCs(state.pressure, state.cfg)
        else:
            self.pressure = np.full_like(self.enthalpy, 0)

        self.temperature = bc.temperature_BCs(state.temperature, state.time, state.cfg)
        self.liquid_salinity = bc.liquid_salinity_BCs(state.liquid_salinity, state.cfg)
        self.dissolved_gas = bc.dissolved_gas_BCs(state.dissolved_gas, state.cfg)
        self.gas_fraction = bc.gas_fraction_BCs(state.gas_fraction, state.cfg)
        self.liquid_fraction = bc.liquid_fraction_BCs(state.liquid_fraction, state.cfg)

    def calculate_brine_convection_sink(self):
        heat_sink = calculate_heat_sink(self, self.cfg)
        salt_sink = calculate_salt_sink(self, self.cfg)
        gas_sink = calculate_gas_sink(self, self.cfg)
        return np.hstack((heat_sink, salt_sink, gas_sink))

    def calculate_dz_fluxes(self, Wl, Vg, V, D_g, D_e):
        heat_flux = calculate_heat_flux(self, Wl, V, D_g, self.cfg)
        salt_flux = calculate_salt_flux(self, Wl, V, D_g, self.cfg)
        gas_flux = calculate_gas_flux(self, Wl, V, Vg, D_g, self.cfg)
        dz = lambda flux: np.matmul(D_e, flux)
        return np.hstack((dz(heat_flux), dz(salt_flux), dz(gas_flux)))
