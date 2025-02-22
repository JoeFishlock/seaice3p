from functools import partial
from typing import Tuple
from dataclasses import dataclass
from serde import serde, coerce
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from .forcing import _filter_missing_values
from .dimensional import (
    DimensionalParams,
    DimensionalFixedTempOceanForcing,
    DimensionalFixedHeatFluxOceanForcing,
    DimensionalBRW09OceanForcing,
    DimensionalMonthlyHeatFluxOceanForcing,
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
    """Provides constant dimensionless ocean heat flux at the bottom of the domain and fixed gas
    saturation state."""

    ocean_heat_flux: float = 1
    ocean_gas_sat: float = 1.0


@serde(type_check=coerce)
class MonthlyHeatFluxOceanForcing:
    """Provides constant dimensionless ocean heat flux at the bottom of the domain in
    each month

    and ocean gas saturation state.

    Proivde an average monthly ocean heat flux with the entries
    i=0, 1, 2, 3, ...., 11
    in the tuple corresponding to the months
    January, February, March, April, ...., December

    Args:
        monthly_ocean_heat_flux: Tuple of dimensionless ocean heat flux values in
        each month
    """

    start_date: str
    timescale_in_days: float
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
    ocean_gas_sat: float = 1.0

    def __post_init__(self):
        # Provide functions to interpolate day of year to monthly ocean heat flux
        self._interpolate_heat_flux = partial(
            np.interp,
            xp=np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]),
            fp=np.array(self.monthly_ocean_heat_flux),
            period=365,
        )

    def get_ocean_heat_flux(self, simulation_time: float) -> float:
        start_datetime = datetime.strptime(self.start_date, "%Y-%m-%d")
        current_datetime = start_datetime + timedelta(
            days=self.timescale_in_days * simulation_time
        )
        current_day = (
            current_datetime - datetime(start_datetime.year, 1, 1)
        ).total_seconds() / 86400
        return self._interpolate_heat_flux(current_day)


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
    FixedTempOceanForcing
    | FixedHeatFluxOceanForcing
    | BRW09OceanForcing
    | MonthlyHeatFluxOceanForcing
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
        case DimensionalMonthlyHeatFluxOceanForcing():
            monthly_ocean_heat_flux = tuple(
                [
                    scales.convert_from_dimensional_heat_flux(ocean_heat_flux)
                    for ocean_heat_flux in dimensional_params.ocean_forcing_config.monthly_ocean_heat_flux
                ]
            )
            return MonthlyHeatFluxOceanForcing(
                start_date=dimensional_params.forcing_config.start_date,
                timescale_in_days=dimensional_params.scales.time_scale,
                monthly_ocean_heat_flux=monthly_ocean_heat_flux,
                ocean_gas_sat=ocean_gas_sat,
            )

        case DimensionalBRW09OceanForcing():
            return BRW09OceanForcing(ocean_gas_sat=ocean_gas_sat)
        case _:
            raise NotImplementedError
