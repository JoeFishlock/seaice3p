"""Template for a solver
concrete solvers should inherit and overwrite required methods"""

import numpy as np
from abc import ABC, abstractmethod
import celestine.params as cp
import celestine.grids as grids
import celestine.boundary_conditions as bc
import celestine.logging_config as logs


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

    def generate_storage_arrays(self, enthalpy, salt, gas, pressure):
        stored_enthalpy = np.copy(enthalpy)
        stored_salt = np.copy(salt)
        stored_gas = np.copy(gas)
        stored_pressure = np.copy(pressure)
        stored_times = np.array([0])
        return stored_times, stored_enthalpy, stored_salt, stored_gas, stored_pressure

    def save_storage(
        self,
        stored_times,
        stored_enthalpy,
        stored_salt,
        stored_gas,
        stored_pressure,
    ):
        data_path = self.cfg.data_path
        name = self.cfg.name
        np.savez(
            f"{data_path}{name}.npz",
            times=stored_times,
            enthalpy=np.transpose(stored_enthalpy),
            salt=np.transpose(stored_salt),
            gas=np.transpose(stored_gas),
            pressure=np.transpose(stored_pressure),
        )

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

        (
            stored_times,
            stored_enthalpy,
            stored_salt,
            stored_gas,
            stored_pressure,
        ) = self.generate_storage_arrays(enthalpy, salt, gas, pressure)
        time_to_save = 0
        time = 0
        while time < T:
            enthalpy, salt, gas, pressure, time, timestep, min_timestep = self.advance(
                enthalpy, salt, gas, pressure, time, timestep
            )
            time_to_save += timestep
            print(f"time={time:.3f}/{T}, timestep={timestep:.2g} \r", end="")
            if np.min(salt) < -self.cfg.physical_params.concentration_ratio:
                raise ValueError("salt crash")

            if self.cfg.numerical_params.adaptive_timestepping:
                timestep = min_timestep

            if (time_to_save - self.cfg.savefreq) >= 0:
                time_to_save = 0
                stored_times = np.append(stored_times, time)
                stored_enthalpy = np.vstack((stored_enthalpy, enthalpy))
                stored_salt = np.vstack((stored_salt, salt))
                stored_gas = np.vstack((stored_gas, gas))
                stored_pressure = np.vstack((stored_pressure, pressure))

        self.save_storage(
            stored_times, stored_enthalpy, stored_salt, stored_gas, stored_pressure
        )
        # clear line after carriage return
        print("")
        return 0
