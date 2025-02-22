from dataclasses import dataclass
from serde import serde, coerce
from typing import Tuple


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalFixedTempOceanForcing:
    """Fixed temperature and gas saturation ocean boundary condition"""

    ocean_temp: float = -1


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalFixedHeatFluxOceanForcing:
    """Provides constant ocean heat flux at the bottom of the domain

    Args:
        ocean_heat_flux: The constant heat flux at the bottom of the domain in W/m2
    """

    ocean_heat_flux: float = 1


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalMonthlyHeatFluxOceanForcing:
    """Provides constant ocean heat flux at the bottom of the domain in each month

    Proivde an average monthly ocean heat flux with the entries
    i=0, 1, 2, 3, ...., 11
    in the tuple corresponding to the months
    January, February, March, April, ...., December

    Args:
        monthly_ocean_heat_flux: Tuple of ocean heat flux values in each month in W/m2
    """

    monthly_ocean_heat_flux: Tuple[float, ...] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalBRW09OceanForcing:
    """Ocean temperature provided by Barrow 2009 data at 2.4m and specify ocean
    fixed gas saturation state"""

    pass
