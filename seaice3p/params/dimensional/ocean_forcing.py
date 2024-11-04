from dataclasses import dataclass
from serde import serde, coerce


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
class DimensionalBRW09OceanForcing:
    """Ocean temperature provided by Barrow 2009 data at 2.4m and specify ocean
    fixed gas saturation state"""

    pass
