from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass(frozen=True)
class DISEQStateBCs:
    """Stores information needed for solution at one timestep with BCs on ghost
    cells as well

    Initialiase the prime variables for the solver:
    enthalpy, bulk salinity and bulk air
    """

    time: float
    enthalpy: NDArray
    salt: NDArray

    temperature: NDArray
    liquid_salinity: NDArray
    dissolved_gas: NDArray
    liquid_fraction: NDArray
    bulk_dissolved_gas: NDArray
    gas_fraction: NDArray
