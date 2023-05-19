"""Template for a solver
concrete solvers should inherit and overwrite required methods"""

import numpy as np
from abc import ABC, abstractmethod
import celestine.params as cp
import celestine.grids as grids
import celestine.logging_config as logs
from celestine.state import State, Solution


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
        chi = self.cfg.physical_params.expansion_coefficient

        bottom_temp = self.cfg.boundary_conditions_config.far_temp
        bottom_bulk_salinity = self.cfg.boundary_conditions_config.far_bulk_salinity
        bottom_dissolved_gas = self.cfg.boundary_conditions_config.far_gas_sat
        bottom_bulk_gas = bottom_dissolved_gas * chi

        # Initialise uniform enthalpy assuming completely liquid initial domain
        enthalpy = np.full((self.I,), bottom_temp)
        salt = np.full_like(enthalpy, bottom_bulk_salinity)
        gas = np.full_like(enthalpy, bottom_bulk_gas)
        pressure = np.full_like(enthalpy, 0)

        initial_state = State(self.cfg, 0, enthalpy, salt, gas, pressure)

        return initial_state

    @abstractmethod
    def take_timestep(self, state: State) -> State:
        """advance enthalpy, salt, gas and pressure to the next timestep.

        Note as of 2023-05-17 removed ability to have adaptive timestepping for
        simplicity.

        :param state: object containing current time, enthalpy, salt, gas, pressure
            and surface temperature.
        :type state: ``celestine.solvers.template.State``
        :return: state of system (new enthalpy, salt, gas and pressure) after one
            timestep.
        """
        pass

    def pre_solve_checks(self):
        """Optionally implement this method if you want to check anything before
        running the solver.

        For example to check the timestep and grid step satisfy some constraint.
        """
        pass

    @logs.time_function
    def solve(self):
        self.pre_solve_checks()  # optional method
        state = self.generate_initial_solution()
        T = self.cfg.total_time
        timestep = self.cfg.numerical_params.timestep

        solution = Solution(self.cfg)
        solution.add_state(state, 0)

        old_time_index = 0

        while state.time < T:
            state = self.take_timestep(state)
            new_time_index = int(state.time / self.cfg.savefreq)

            print(
                f"{self.cfg.name}: time={state.time:.3f}/{T}, timestep={timestep:.2g} \r",
                end="",
            )

            if np.min(state.salt) < -self.cfg.physical_params.concentration_ratio:
                raise ValueError("salt crash")

            if new_time_index - old_time_index > 0:
                solution.add_state(state, index=new_time_index)

            old_time_index = new_time_index

        solution.save()
        # clear line after carriage return
        print("")
        return 0
