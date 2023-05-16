"""Template for a solver
concrete solvers should inherit and overwrite required methods"""

import numpy as np
from abc import ABC, abstractmethod
import celestine.params as cp
import celestine.grids as grids
import celestine.boundary_conditions as bc
import celestine.logging_config as logs

from celestine.enthalpy_method import calculate_enthalpy_method
from celestine.phase_boundaries import get_phase_masks


class State:
    """Stores information needed for solution at one timestep"""

    def __init__(
        self, time, enthalpy, salt, gas, pressure=None, top_temperature=np.NaN
    ):
        self.time = time
        self.enthalpy = enthalpy
        self.salt = salt
        self.gas = gas
        self.top_temperature = top_temperature

        if pressure is not None:
            self.pressure = pressure
        else:
            self.pressure = np.full_like(self.enthalpy, 0)

    def calculate_enthalpy_method(self, cfg):
        phase_masks = get_phase_masks(self.enthalpy, self.salt, self.gas, cfg)
        (
            temperature,
            liquid_fraction,
            gas_fraction,
            solid_fraction,
            liquid_salinity,
            dissolved_gas,
        ) = calculate_enthalpy_method(
            self.enthalpy, self.salt, self.gas, cfg, phase_masks
        )
        self.temperature = temperature
        self.liquid_fraction = liquid_fraction
        self.gas_fraction = gas_fraction
        self.solid_fraction = solid_fraction
        self.liquid_salinity = liquid_salinity
        self.dissolved_gas = dissolved_gas


class Solution:
    """store solution at specified times on the center grid"""

    def __init__(self, cfg: cp.Config):
        self.time_length = 1 + int(cfg.total_time / cfg.savefreq)
        self.name = cfg.name
        self.data_path = cfg.data_path

        self.times = np.zeros((self.time_length,))
        self.top_temperature = np.zeros_like(self.times)

        self.enthalpy = np.zeros((cfg.numerical_params.I, self.time_length))
        self.salt = np.zeros_like(self.enthalpy)
        self.gas = np.zeros_like(self.enthalpy)
        self.pressure = np.zeros_like(self.enthalpy)

    def add_state(self, state: State, index: int):
        """add state to stored solution at given time index"""
        self.times[index] = state.time
        self.top_temperature[index] = state.top_temperature
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


class SolverTemplate(ABC):
    def __init__(self, cfg: cp.Config):
        """initialise solver object

        Assign step size, number of cells and difference matrices for convenience.

        :param cfg: simulation configuration
        """
        self.cfg = cfg
        self.step = cfg.numerical_params.step
        self.I = cfg.numerical_params.I
        self.D_e = grids.get_difference_matrix(self.I, self.step)
        self.D_g = grids.get_difference_matrix(self.I + 1, self.step)

    def generate_initial_solution(self):
        """Generate initial solution on the ghost grid

        :returns: initial solution arrays on ghost grid (enthalpy, salt, gas, pressure)
        """
        ghost_length = self.I + 2
        C = self.cfg.physical_params.concentration_ratio
        chi = self.cfg.physical_params.expansion_coefficient
        bottom_dissolved_gas = self.cfg.boundary_conditions_config.far_gas_sat

        bottom_bulk_salinity = C
        bottom_temp = self.cfg.boundary_conditions_config.far_temp
        bottom_bulk_gas = bottom_dissolved_gas * chi

        bottom_enthalpy = bc.calculate_enthalpy_from_temp(
            bottom_bulk_salinity,
            bottom_bulk_gas,
            bottom_temp,
            self.cfg,
        )
        enthalpy = np.full((ghost_length,), bottom_enthalpy)
        salt = np.full_like(enthalpy, 0)
        gas = np.full_like(
            enthalpy,
            bottom_bulk_gas,
        )
        pressure = np.full_like(enthalpy, 0)
        return enthalpy, salt, gas, pressure

    @abstractmethod
    def take_timestep(self, enthalpy, salt, gas, pressure, time, timestep):
        """advance enthalpy, salt, gas and pressure to the next timestep.

        Allowing for the possibiltiy of a variable timestep we make this an input param
        and return the minimal allowable timestep from stability criteria.

        :param enthalpy:
        :param salt:
        :param gas:
        :param pressure:
        :param time:
        :param timestep:

        :return: (new_enthalpy, new_salt, new_gas, new_pressure, new_time, min_timestep)
        """
        pass

    def advance(self, enthalpy, salt, gas, pressure, time, timestep):
        (
            new_enthalpy,
            new_salt,
            new_gas,
            new_pressure,
            new_time,
            min_timestep,
        ) = self.take_timestep(enthalpy, salt, gas, pressure, time, timestep)
        if self.cfg.numerical_params.adaptive_timestepping:
            while timestep > min_timestep:
                timestep = min_timestep
                (
                    new_enthalpy,
                    new_salt,
                    new_gas,
                    new_pressure,
                    new_time,
                    min_timestep,
                ) = self.take_timestep(enthalpy, salt, gas, pressure, time, timestep)

        return (
            new_enthalpy,
            new_salt,
            new_gas,
            new_pressure,
            new_time,
            timestep,
            min_timestep,
        )

    @logs.time_function
    def solve(self):
        enthalpy, salt, gas, pressure = self.generate_initial_solution()
        T = self.cfg.total_time
        timestep = self.cfg.numerical_params.timestep

        solution = Solution(self.cfg)
        initial_state = State(0, enthalpy[1:-1], salt[1:-1], gas[1:-1], pressure[1:-1])
        solution.add_state(initial_state, 0)

        time = 0
        old_time_index = 0
        while time < T:
            enthalpy, salt, gas, pressure, time, timestep, min_timestep = self.advance(
                enthalpy, salt, gas, pressure, time, timestep
            )
            new_time_index = int(time / self.cfg.savefreq)

            print(f"time={time:.3f}/{T}, timestep={timestep:.2g} \r", end="")
            if np.min(salt) < -self.cfg.physical_params.concentration_ratio:
                raise ValueError("salt crash")

            if self.cfg.numerical_params.adaptive_timestepping:
                timestep = min_timestep

            if new_time_index - old_time_index > 0:
                time_to_save = 0
                state = State(
                    time, enthalpy[1:-1], salt[1:-1], gas[1:-1], pressure[1:-1]
                )
                solution.add_state(state, index=new_time_index)

            old_time_index = new_time_index

        solution.save()
        # clear line after carriage return
        print("")
        return 0
