from abc import ABC, abstractmethod, abstractclassmethod
import celestine.params as cp
from celestine.grids import Grids


class State(ABC):
    """Stores information needed for solution at one timestep on cell centers"""

    @abstractmethod
    def __init__(self, cfg: cp.Config):
        self.cfg = cfg
        pass

    @property
    def grid(self):
        return Grids(self.cfg.numerical_params.I).centers
