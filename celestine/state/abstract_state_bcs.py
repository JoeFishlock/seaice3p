"""Classes to store solution variables

State: store variables on cell centers
StateBCs: add boundary conditions in ghost cells to cell center variables
Solution: store primary variables at each timestep we want to save data
"""

from abc import ABC, abstractmethod
from celestine.grids import initialise_grids
from .abstract_state import State


class StateBCs(ABC):
    """Stores information needed for solution at one timestep with BCs on ghost
    cells as well

    Note must initialise once enthalpy method has already run on State."""

    @abstractmethod
    def __init__(self, state: State):
        self.cfg = state.cfg

    @property
    def grid(self):
        _, _, _, ghosts = initialise_grids(self.cfg.numerical_params.I)
        return ghosts

    @property
    def edge_grid(self):
        _, _, edges, _ = initialise_grids(self.cfg.numerical_params.I)
        return edges

    @abstractmethod
    def calculate_brine_convection_sink(self):
        pass

    @abstractmethod
    def calculate_dz_fluxes(self, Wl, Vg, V, D_g, D_e):
        pass
