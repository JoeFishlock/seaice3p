"""Classes to store solution variables

State: store variables on cell centers
StateBCs: add boundary conditions in ghost cells to cell center variables
Solution: store primary variables at each timestep we want to save data
"""

import numpy as np
import celestine.params as cp
import celestine.boundary_conditions as bc
from celestine.enthalpy_method import get_enthalpy_method
from celestine.grids import initialise_grids


class State:
    """Stores information needed for solution at one timestep on cell centers"""

    def __init__(self, cfg: cp.Config, time, enthalpy, salt, gas, pressure=None):
        self.cfg = cfg
        self.time = time
        self.enthalpy = enthalpy
        self.salt = salt
        self.gas = gas

        if pressure is not None:
            self.pressure = pressure
        else:
            self.pressure = np.full_like(self.enthalpy, 0)

    @property
    def grid(self):
        _, centers, _, _ = initialise_grids(self.cfg.numerical_params.I)
        return centers

    def calculate_enthalpy_method(self):
        enthalpy_method = get_enthalpy_method(self.cfg)(self.cfg.physical_params)
        (
            temperature,
            liquid_fraction,
            gas_fraction,
            solid_fraction,
            liquid_salinity,
            dissolved_gas,
        ) = enthalpy_method.calculate_enthalpy_method(self)
        self.temperature = temperature
        self.liquid_fraction = liquid_fraction
        self.gas_fraction = gas_fraction
        self.solid_fraction = solid_fraction
        self.liquid_salinity = liquid_salinity
        self.dissolved_gas = dissolved_gas


class StateBCs:
    """Stores information needed for solution at one timestep with BCs on ghost
    cells as well

    Note must initialise once enthalpy method has already run on State."""

    def __init__(self, state: State):
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

    @property
    def grid(self):
        _, _, _, ghosts = initialise_grids(self.cfg.numerical_params.I)
        return ghosts


class Solution:
    """store solution at specified times on the center grid"""

    def __init__(self, cfg: cp.Config):
        self.time_length = 1 + int(cfg.total_time / cfg.savefreq)
        self.name = cfg.name
        self.data_path = cfg.data_path

        self.times = np.zeros((self.time_length,))

        self.enthalpy = np.zeros((cfg.numerical_params.I, self.time_length))
        self.salt = np.zeros_like(self.enthalpy)
        self.gas = np.zeros_like(self.enthalpy)
        self.pressure = np.zeros_like(self.enthalpy)

    def add_state(self, state: State, index: int):
        """add state to stored solution at given time index"""
        self.times[index] = state.time
        self.enthalpy[:, index] = state.enthalpy
        self.salt[:, index] = state.salt
        self.gas[:, index] = state.gas
        self.pressure[:, index] = state.pressure

    def save(self):
        data_path = self.data_path
        name = self.name
        np.savez(
            f"{data_path}{name}.npz",
            times=self.times,
            enthalpy=self.enthalpy,
            salt=self.salt,
            gas=self.gas,
            pressure=self.pressure,
        )
