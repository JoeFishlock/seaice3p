from functools import cached_property
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import celestine.params as cp
from ..enthalpy_method import DisequilibriumEnthalpyMethod
from .disequilibrium_state_bcs import DISEQStateBCs


@dataclass(frozen=True)
class DISEQState:
    """Contains the principal variables for solution with non-equilibrium gas phase.
    The total bulk gas is partitioned between dissolved gas and free phase gas with
    a finite nucleation rate (non dimensional damkohler number).

    principal solution components:
    bulk enthalpy
    bulk salinity
    bulk dissolved gas
    gas fraction

    all on the center grid.

    Note:
    Define bulk dissolved gas for the system as

    expansion_coefficient * liquid_fraction * dissolved_gas

    so that this is different from the dissolved gas concentration and

    bulk_gas = bulk_dissolved_gas + gas_fraction

    in non-dimensional units.
    """

    time: float
    enthalpy: NDArray
    salt: NDArray
    bulk_dissolved_gas: NDArray
    gas_fraction: NDArray

    @cached_property
    def gas(self) -> NDArray:
        """Calculate bulk gas content and use same attribute name as EQMState"""
        return self.bulk_dissolved_gas + self.gas_fraction


def get_state_with_bcs(self):
    """Initialise the appropriate StateBCs object"""
    return DISEQStateBCs(self)


def init_from_stacked_state(cls, cfg: cp.Config, time, stacked_state):
    """initialise from stacked solution vector for use in the solver"""
    enthalpy, salt, bulk_dissolved_gas, gas_fraction = np.split(stacked_state, 4)
    return cls(cfg, time, enthalpy, salt, bulk_dissolved_gas, gas_fraction)


def get_stacked_state(self):
    return np.hstack(
        (self.enthalpy, self.salt, self.bulk_dissolved_gas, self.gas_fraction)
    )


def calculate_enthalpy_method(self):
    (
        temperature,
        liquid_fraction,
        gas_fraction,
        solid_fraction,
        liquid_salinity,
        dissolved_gas,
    ) = DisequilibriumEnthalpyMethod(
        self.cfg.physical_params
    ).calculate_enthalpy_method(
        self
    )
    self.temperature = temperature
    self.liquid_fraction = liquid_fraction
    self.gas_fraction = gas_fraction
    self.solid_fraction = solid_fraction
    self.liquid_salinity = liquid_salinity
    self.dissolved_gas = dissolved_gas
