from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


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


@dataclass(frozen=True)
class EQMStateFull:
    """Contains all variables variables for solution with equilibrium gas phase
    after running the enthalpy method on EQMSate.

    principal solution components:
    bulk enthalpy
    bulk salinity
    bulk gas

    enthalpy method variables:
    temperature
    liquid_fraction
    solid_fraction
    liquid_salinity
    dissolved_gas
    gas_fraction

    all on the center grid.
    """

    time: float
    enthalpy: NDArray
    salt: NDArray
    gas: NDArray

    temperature: NDArray
    liquid_fraction: NDArray
    solid_fraction: NDArray
    liquid_salinity: NDArray
    dissolved_gas: NDArray
    gas_fraction: NDArray


@dataclass(frozen=True)
class EQMStateBCs:
    """Stores information needed for solution at one timestep with BCs on ghost
    cells as well

    Initialiase the prime variables for the solver:
    enthalpy, bulk salinity and bulk air
    """

    time: float
    enthalpy: NDArray
    salt: NDArray
    gas: NDArray

    temperature: NDArray
    liquid_salinity: NDArray
    dissolved_gas: NDArray
    gas_fraction: NDArray
    liquid_fraction: NDArray
