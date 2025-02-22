from functools import partial
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from numpy.typing import NDArray
from serde import serde, coerce
import numpy as np
from .dimensional import (
    DimensionalParams,
    DimensionalConstantForcing,
    DimensionalBRW09Forcing,
    DimensionalYearlyForcing,
    DimensionalRadForcing,
    DimensionalRobinForcing,
    DimensionalSWForcing,
    DimensionalConstantSWForcing,
    DimensionalOilHeating,
    DimensionalBackgroundOilHeating,
    DimensionalLWForcing,
    DimensionalConstantLWForcing,
    DimensionalTurbulentFlux,
    DimensionalConstantTurbulentFlux,
    DimensionalERA5Forcing,
    ERA5FileKeys,
)
import xarray as xr
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units as metpyunits


def _filter_missing_values(air_temp, days):
    """Filter out missing values are recorded as 9999"""
    is_missing = np.abs(air_temp) > 100
    return air_temp[~is_missing], days[~is_missing]


@serde(type_check=coerce)
@dataclass(frozen=True)
class ConstantForcing:
    """Constant temperature forcing"""

    constant_top_temperature: float = -1.5


@serde(type_check=coerce)
@dataclass(frozen=True)
class YearlyForcing:
    """Yearly sinusoidal temperature forcing"""

    offset: float = -1.0
    amplitude: float = 0.75
    period: float = 4.0


@serde(type_check=coerce)
class BRW09Forcing:
    """Surface and ocean temperature data loaded from thermistor temperature record
    during the Barrow 2009 field study.
    """

    Barrow_top_temperature_data_choice: str = "air"

    def __post_init__(self):
        """populate class attributes with barrow dimensional air temperature
        and time in days (with missing values filtered out).

        Note the metadata explaining how to use the barrow temperature data is also
        in seaice3p/forcing_data. The indices corresponding to days and air temp are
        hard coded in as class variables.
        """
        DATA_INDICES = {
            "time": 0,
            "air": 8,
            "bottom_snow": 18,
            "top_ice": 19,
        }
        data = np.genfromtxt(
            Path(__file__).parent.parent / "forcing_data/BRW09.txt", delimiter="\t"
        )
        top_temp_index = DATA_INDICES[self.Barrow_top_temperature_data_choice]
        time_index = DATA_INDICES["time"]

        barrow_top_temp = data[:, top_temp_index]
        barrow_days = data[:, time_index] - data[0, time_index]
        barrow_top_temp, barrow_days = _filter_missing_values(
            barrow_top_temp, barrow_days
        )

        self.barrow_top_temp = barrow_top_temp
        self.barrow_days = barrow_days


@serde(type_check=coerce)
@dataclass(frozen=True)
class RadForcing:
    """Forcing parameters for radiative transfer simulation with oil drops

    we have not implemented the non-dimensionalisation for these parameters yet
    and so we just pass the dimensional values directly to the simulation"""

    SW_forcing: DimensionalSWForcing = DimensionalConstantSWForcing()
    LW_forcing: DimensionalLWForcing = DimensionalConstantLWForcing()
    turbulent_flux: DimensionalTurbulentFlux = DimensionalConstantTurbulentFlux()
    oil_heating: DimensionalOilHeating = DimensionalBackgroundOilHeating()


@serde(type_check=coerce)
class ERA5Forcing:
    """Forcing parameters for simulation forced with atmospheric variables
    from reanalysis data in netCDF file located at data_path.

    Never create this object directly but instead initialise from a dimensional
    simulation configuration as we must pass it the simulation timescale to correctly
    read the atmospheric variables from the netCDF file.
    """

    data_path: Path
    start_date: str
    timescale_in_days: float
    forcing_data_file_keys: ERA5FileKeys = ERA5FileKeys()
    snow_density: Optional[float] = None
    SW_forcing: DimensionalSWForcing = DimensionalConstantSWForcing()
    LW_forcing: DimensionalLWForcing = DimensionalConstantLWForcing()
    turbulent_flux: DimensionalTurbulentFlux = DimensionalConstantTurbulentFlux()
    oil_heating: DimensionalOilHeating = DimensionalBackgroundOilHeating()

    def __post_init__(self):
        data = xr.open_dataset(self.data_path)
        DATES = getattr(data, self.forcing_data_file_keys.time).to_numpy()
        DIMLESS_TIMES = (1 / self.timescale_in_days) * np.array(
            [
                (date - np.datetime64(self.start_date)) / np.timedelta64(1, "D")
                for date in DATES
            ]
        )

        # convert to deg C
        T2M = (
            getattr(data, self.forcing_data_file_keys.temperature_at_2m_in_K).to_numpy()
            - 273.15
        )
        D2M = (
            getattr(data, self.forcing_data_file_keys.dewpoint_at_2m_in_K).to_numpy()
            - 273.15
        )

        LW = getattr(
            data, self.forcing_data_file_keys.longwave_radiation_in_W_m2
        ).to_numpy()
        SW = getattr(
            data, self.forcing_data_file_keys.shortwave_radiation_in_W_m2
        ).to_numpy()

        # convert to KPa
        ATM = (
            getattr(data, self.forcing_data_file_keys.surface_pressure_in_Pa).to_numpy()
            / 1e3
        )

        wind_key = self.forcing_data_file_keys.windspeed_at_2m_in_m_s
        if wind_key is None:
            WIND = np.full_like(DIMLESS_TIMES, self.turbulent_flux.windspeed)
        else:
            WIND = getattr(data, wind_key).to_numpy()

        # Calculate specific humidity in kg/kg from dewpoint temperature
        SPEC_HUM = _calculate_specific_humidity(ATM, D2M)

        snow_key = self.forcing_data_file_keys.snow_depth_in_m
        # if ERA5 standard short name for snow depth in m of water equivalent use snow
        # density to convert to m of snow
        if snow_key == "sd":
            if self.snow_density is None:
                raise ValueError("No snow density provided")
            SNOW_DEPTH = getattr(data, "sd").to_numpy() * (1000 / self.snow_density)

        # If snow key is another name assume snow depth is just in m of snow
        elif snow_key is not None:
            SNOW_DEPTH = getattr(data, snow_key).to_numpy()

        # If snow key is None assume no snow
        else:
            SNOW_DEPTH = np.zeros_like(DIMLESS_TIMES)

        # Provide functions to interpolate forcing data at non-dimensional times
        # during simulation
        self.get_2m_temp = partial(
            np.interp, xp=DIMLESS_TIMES, fp=T2M, left=np.nan, right=np.nan
        )
        self.get_LW = partial(
            np.interp, xp=DIMLESS_TIMES, fp=LW, left=np.nan, right=np.nan
        )
        self.get_SW = partial(
            np.interp, xp=DIMLESS_TIMES, fp=SW, left=np.nan, right=np.nan
        )
        self.get_ATM = partial(
            np.interp, xp=DIMLESS_TIMES, fp=ATM, left=np.nan, right=np.nan
        )
        self.get_spec_hum = partial(
            np.interp, xp=DIMLESS_TIMES, fp=SPEC_HUM, left=np.nan, right=np.nan
        )
        self.get_snow_depth = partial(
            np.interp, xp=DIMLESS_TIMES, fp=SNOW_DEPTH, left=np.nan, right=np.nan
        )
        self.get_windspeed = partial(
            np.interp, xp=DIMLESS_TIMES, fp=WIND, left=np.nan, right=np.nan
        )


def _calculate_specific_humidity(pressure: NDArray, dewpoint: NDArray) -> NDArray:
    """Take ERA5 data and return specific humidity at 2m in kg/kg"""
    return (
        specific_humidity_from_dewpoint(
            pressure * metpyunits.kPa, dewpoint * metpyunits.degC
        )
        .to("kg/kg")
        .magnitude
    )


@serde(type_check=coerce)
@dataclass(frozen=True)
class RobinForcing:
    """Dimensionless forcing parameters for Robin boundary condition"""

    biot: float = 12
    restoring_temperature: float = -1.3


ForcingConfig = (
    ConstantForcing
    | YearlyForcing
    | BRW09Forcing
    | RadForcing
    | RobinForcing
    | ERA5Forcing
)


def get_dimensionless_forcing_config(
    dimensional_params: DimensionalParams,
) -> ForcingConfig:
    scales = dimensional_params.scales
    match dimensional_params.forcing_config:
        case DimensionalConstantForcing():
            top_temp = scales.convert_from_dimensional_temperature(
                dimensional_params.forcing_config.constant_top_temperature
            )
            return ConstantForcing(
                constant_top_temperature=top_temp,
            )
        case DimensionalYearlyForcing():
            return YearlyForcing(
                offset=dimensional_params.forcing_config.offset,
                amplitude=dimensional_params.forcing_config.amplitude,
                period=dimensional_params.forcing_config.period,
            )
        case DimensionalBRW09Forcing():
            return BRW09Forcing(
                Barrow_top_temperature_data_choice=dimensional_params.forcing_config.Barrow_top_temperature_data_choice,
            )
        case DimensionalRadForcing():
            return RadForcing(
                SW_forcing=dimensional_params.forcing_config.SW_forcing,
                LW_forcing=dimensional_params.forcing_config.LW_forcing,
                turbulent_flux=dimensional_params.forcing_config.turbulent_flux,
                oil_heating=dimensional_params.forcing_config.oil_heating,
            )
        case DimensionalRobinForcing():
            restoring_temperature = scales.convert_from_dimensional_temperature(
                dimensional_params.forcing_config.restoring_temperature
            )
            biot = (
                dimensional_params.lengthscale
                * dimensional_params.forcing_config.heat_transfer_coefficient
                / dimensional_params.water_params.liquid_thermal_conductivity
            )
            return RobinForcing(
                biot=biot,
                restoring_temperature=restoring_temperature,
            )
        case DimensionalERA5Forcing():
            return ERA5Forcing(
                data_path=dimensional_params.forcing_config.data_path,
                start_date=dimensional_params.forcing_config.start_date,
                timescale_in_days=dimensional_params.scales.time_scale,
                forcing_data_file_keys=dimensional_params.forcing_config.forcing_data_file_keys,
                snow_density=dimensional_params.water_params.snow_density,
                SW_forcing=dimensional_params.forcing_config.SW_forcing,
                LW_forcing=dimensional_params.forcing_config.LW_forcing,
                turbulent_flux=dimensional_params.forcing_config.turbulent_flux,
                oil_heating=dimensional_params.forcing_config.oil_heating,
            )
        case _:
            raise NotImplementedError
