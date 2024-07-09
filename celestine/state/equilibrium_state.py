from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import celestine.params as cp
from celestine.enthalpy_method import ReducedEnthalpyMethod
from .equilibrium_state_bcs import EQMStateBCs


@dataclass(frozen=True)
class EQMState:
    """Contains the principal variables for solution with equilibrium gas phase:

    bulk enthalpy
    bulk salinity
    bulk gas

    all on the center grid.
    """

    time: float
    enthalpy: NDArray
    salt: NDArray
    gas: NDArray


def get_state_with_bcs(state: EQMState) -> EQMStateBCs:
    """Initialise the appropriate StateBCs object"""
    raise NotImplementedError


def init_from_stacked_state(cls, cfg: cp.Config, time, stacked_state):
    """initialise from stacked solution vector for use in the solver"""
    enthalpy, salt, gas = np.split(stacked_state, 3)
    return cls(cfg, time, enthalpy, salt, gas)


def get_stacked_state(self):
    return np.hstack((self.enthalpy, self.salt, self.gas))


def calculate_enthalpy_method(self):
    (
        temperature,
        liquid_fraction,
        gas_fraction,
        solid_fraction,
        liquid_salinity,
        dissolved_gas,
    ) = ReducedEnthalpyMethod(self.cfg.physical_params).calculate_enthalpy_method(self)
    self.temperature = temperature
    self.liquid_fraction = liquid_fraction
    self.gas_fraction = gas_fraction
    self.solid_fraction = solid_fraction
    self.liquid_salinity = liquid_salinity
    self.dissolved_gas = dissolved_gas
