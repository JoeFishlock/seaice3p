from dataclasses import dataclass
from serde import serde, coerce
from pathlib import Path
import numpy as np
from .forcing import _filter_missing_values
from .dimensional import (
    DimensionalParams,
    DimensionalFixedTempOceanForcing,
    DimensionalFixedHeatFluxOceanForcing,
    DimensionalBRW09OceanForcing,
)


@serde(type_check=coerce)
@dataclass(frozen=True)
class FixedTempOceanForcing:
    """Fixed temperature and gas saturation ocean boundary condition"""

    ocean_temp: float = 0.1
    ocean_gas_sat: float = 1.0


@serde(type_check=coerce)
@dataclass(frozen=True)
class FixedHeatFluxOceanForcing:
    """Provides constant ocean heat flux at the bottom of the domain and fixed gas
    saturation state."""

    ocean_heat_flux: float = 1  # W /m2
    ocean_gas_sat: float = 1.0


@serde(type_check=coerce)
class BRW09OceanForcing:
    """Ocean temperature provided by Barrow 2009 data at 2.4m and specify ocean
    fixed gas saturation state"""

    ocean_gas_sat: float = 1.0

    def __post_init__(self):
        """populate class attributes with barrow dimensional ocean temperature
        and time in days (with missing values filtered out).

        Note the metadata explaining how to use the barrow temperature data is also
        in seaice3p/forcing_data.
        """
        data = np.genfromtxt(
            Path(__file__).parent.parent / "forcing_data/BRW09.txt", delimiter="\t"
        )
        ocean_temp_index = 43
        time_index = 0

        barrow_bottom_temp = data[:, ocean_temp_index]
        barrow_ocean_days = data[:, time_index] - data[0, time_index]
        barrow_bottom_temp, barrow_ocean_days = _filter_missing_values(
            barrow_bottom_temp, barrow_ocean_days
        )

        self.barrow_bottom_temp = barrow_bottom_temp
        self.barrow_ocean_days = barrow_ocean_days


OceanForcingConfig = (
    FixedTempOceanForcing | FixedHeatFluxOceanForcing | BRW09OceanForcing
)


def get_dimensionless_ocean_forcing_config(
    dimensional_params: DimensionalParams,
) -> OceanForcingConfig:
    ocean_gas_sat = dimensional_params.gas_params.ocean_saturation_state
    scales = dimensional_params.scales
    match dimensional_params.ocean_forcing_config:
        case DimensionalFixedTempOceanForcing():
            ocean_temp = scales.convert_from_dimensional_temperature(
                dimensional_params.ocean_forcing_config.ocean_temp
            )
            return FixedTempOceanForcing(
                ocean_temp=ocean_temp, ocean_gas_sat=ocean_gas_sat
            )
        case DimensionalFixedHeatFluxOceanForcing():
            ocean_heat_flux = scales.convert_from_dimensional_heat_flux(
                dimensional_params.ocean_forcing_config.ocean_heat_flux
            )
            return FixedHeatFluxOceanForcing(
                ocean_heat_flux=ocean_heat_flux, ocean_gas_sat=ocean_gas_sat
            )

        case DimensionalBRW09OceanForcing():
            return BRW09OceanForcing(ocean_gas_sat=ocean_gas_sat)
        case _:
            raise NotImplementedError
