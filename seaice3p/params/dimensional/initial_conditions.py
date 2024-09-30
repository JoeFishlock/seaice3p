from dataclasses import dataclass
from pathlib import Path
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
class DimensionalOilInitialConditions:
    # Parameters for summer initial conditions
    initial_ice_depth: float = 1  # in m
    initial_ocean_temperature: float = -2  # in deg C
    initial_ice_temperature: float = -4  # in deg C
    initial_oil_volume_fraction: float = 1e-7
    initial_ice_bulk_salinity: float = 5.92  # in g/kg
    initial_oil_free_depth: float = 0  # in m


@serde(type_check=coerce)
@dataclass(frozen=True)
class PreviousSimulation:
    data_path: Path
