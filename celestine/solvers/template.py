"""Template for a solver
concrete solvers should inherit and overwrite required methods"""

from pathlib import Path
from abc import ABC, abstractmethod
import celestine.params as cp
import celestine.grids as grids
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

    def pre_solve_checks(self):
        """Optionally implement this method if you want to check anything before
        running the solver.

        For example to check the timestep and grid step satisfy some constraint.
        """
        pass

    def load_forcing_data_if_needed(self):
        if self.cfg.forcing_config.temperature_forcing_choice == "barrow_2009":
            self.cfg.forcing_config.load_forcing_data()

    @abstractmethod
    def solve(self, directory: Path):
        pass
