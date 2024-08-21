"""Classes to store solution variables

State: store variables on cell centers
StateBCs: add boundary conditions in ghost cells to cell center variables
Solution: store primary variables at each timestep we want to save data
"""

from abc import ABC, abstractmethod
from celestine.grids import Grids


class StateBCs(ABC):
    """Stores information needed for solution at one timestep with BCs on ghost
    cells as well

    Note must initialise once enthalpy method has already run on State."""

    @property
    def grid(self):
        return Grids(self.cfg.numerical_params.I).ghosts

    @property
    def edge_grid(self):
        return Grids(self.cfg.numerical_params.I).edges

    @abstractmethod
    def calculate_equation(self, D_g, D_e):
        pass
