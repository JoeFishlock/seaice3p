from dataclasses import dataclass
from serde import serde, coerce


@serde(type_check=coerce)
@dataclass(frozen=True)
class UniformInitialConditions:
    """values for bottom (ocean) boundary"""


@serde(type_check=coerce)
@dataclass(frozen=True)
class BRW09InitialConditions:
    """values for bottom (ocean) boundary"""

    Barrow_initial_bulk_gas_in_ice: float = 1 / 5


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalSummerInitialConditions:
    # Parameters for summer initial conditions
    initial_summer_ice_depth: float = 1  # in m
    initial_summer_ocean_temperature: float = -2  # in deg C
    initial_summer_ice_temperature: float = -4  # in deg C
