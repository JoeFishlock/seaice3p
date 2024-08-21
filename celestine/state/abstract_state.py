from abc import ABC, abstractmethod, abstractclassmethod
import celestine.params as cp
from celestine.grids import Grids


class State(ABC):
    """Stores information needed for solution at one timestep on cell centers"""

    @abstractmethod
    def __init__(self, cfg: cp.Config):
        self.cfg = cfg
        pass

    @abstractmethod
    def get_state_with_bcs(self):
        """Initialise the appropriate StateBCs object"""
        pass

    @property
    def grid(self):
        return Grids(self.cfg.numerical_params.I).centers

    @abstractclassmethod
    def init_from_stacked_state(cls):
        pass

    @abstractmethod
    def calculate_enthalpy_method(self):
        pass
