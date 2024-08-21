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


@dataclass(frozen=True)
class DISEQStateFull:
    """Contains all variables variables for solution with non-equilibrium gas phase
    after running the enthalpy method on DISEQSate.
    The total bulk gas is partitioned between dissolved gas and free phase gas with
    a finite nucleation rate (non dimensional damkohler number).

    principal solution components:
    bulk enthalpy
    bulk salinity
    bulk dissolved gas
    gas fraction

    enthalpy method variables:
    temperature
    liquid_fraction
    solid_fraction
    liquid_salinity
    dissolved_gas

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

    temperature: NDArray
    liquid_fraction: NDArray
    solid_fraction: NDArray
    liquid_salinity: NDArray
    dissolved_gas: NDArray

    @cached_property
    def gas(self) -> NDArray:
        """Calculate bulk gas content and use same attribute name as EQMState"""
        return self.bulk_dissolved_gas + self.gas_fraction


def get_state_with_bcs(full_state: DISEQStateFull):
    """Initialise the appropriate StateBCs object"""
    return DISEQStateBCs(full_state)


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
