"""Template for a solver
concrete solvers should inherit and overwrite required methods"""

import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import celestine.params as cp
import celestine.grids as grids
import celestine.logging_config as logs
from celestine.state import State, Solution, StateBCs
from celestine.initial_conditions import get_initial_conditions


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
        return get_initial_conditions(self.cfg)

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

    def load_forcing_data_if_needed(self):
        if self.cfg.forcing_config.temperature_forcing_choice == "barrow_2009":
            self.cfg.forcing_config.load_forcing_data()

    @logs.time_function
    def solve(self, directory: Path):
        self.pre_solve_checks()  # optional method

        # for the barrow forcing you need to load external data to the forcing config
        self.load_forcing_data_if_needed()

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

        solution.save(directory)
        # clear line after carriage return
        print("")
        return 0


def prevent_gas_rise_into_saturated_cell(Vg, state_BCs: StateBCs):
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
