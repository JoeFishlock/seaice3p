from pathlib import Path
from dataclasses import dataclass
from typing import ClassVar
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
)
import xarray as xr
from scipy.interpolate import CubicSpline
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
    use_snow_data: bool = False
    SW_forcing: DimensionalSWForcing = DimensionalConstantSWForcing()
    LW_forcing: DimensionalLWForcing = DimensionalConstantLWForcing()
    turbulent_flux: DimensionalTurbulentFlux = DimensionalConstantTurbulentFlux()
    oil_heating: DimensionalOilHeating = DimensionalBackgroundOilHeating()

    NEGLIGIBLE_SNOW_DEPTH: ClassVar[
        float
    ] = 0.05  # snow depth in m below which we assume snow is negligible

    def __post_init__(self):
        data = xr.open_dataset(self.data_path)
        daily_data = data.resample(valid_time="1d").mean()
        DATES = daily_data.valid_time.to_numpy()
        DIMLESS_TIMES = (1 / self.timescale_in_days) * np.array(
            [
                (date - np.datetime64(self.start_date)) / np.timedelta64(1, "D")
                for date in DATES
            ]
        )

        # convert to deg C
        T2M = daily_data.t2m[:, 0, 0].to_numpy() - 273.15
        D2M = daily_data.d2m[:, 0, 0].to_numpy() - 273.15

        LW = daily_data.msdwlwrf[:, 0, 0].to_numpy()
        SW = daily_data.msdwswrf[:, 0, 0].to_numpy()

        # convert to KPa
        ATM = daily_data.sp[:, 0, 0].to_numpy() / 1e3

        # Calculate specific humidity in kg/kg from dewpoint temperature
        SPEC_HUM = _calculate_specific_humidity(ATM, D2M)

        self.get_2m_temp = CubicSpline(DIMLESS_TIMES, T2M, extrapolate=False)
        self.get_LW = CubicSpline(DIMLESS_TIMES, LW, extrapolate=False)
        self.get_SW = CubicSpline(DIMLESS_TIMES, SW, extrapolate=False)
        self.get_ATM = CubicSpline(DIMLESS_TIMES, ATM, extrapolate=False)
        self.get_spec_hum = CubicSpline(DIMLESS_TIMES, SPEC_HUM, extrapolate=False)

        if self.use_snow_data:
            SNOW_DENSITY = 100  # kg/m3
            WATER_DENSITY = 1000  # kg/m3

            # Set to zero if less than negligible snow depth
            SNOW_DEPTH = np.maximum(
                daily_data.sd[:, 0, 0].to_numpy() * (WATER_DENSITY / SNOW_DENSITY)
                - self.NEGLIGIBLE_SNOW_DEPTH,
                0,
            )
            self.get_snow_depth = CubicSpline(
                DIMLESS_TIMES, SNOW_DEPTH, extrapolate=False
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
                use_snow_data=dimensional_params.forcing_config.use_snow_data,
                SW_forcing=dimensional_params.forcing_config.SW_forcing,
                LW_forcing=dimensional_params.forcing_config.LW_forcing,
                turbulent_flux=dimensional_params.forcing_config.turbulent_flux,
                oil_heating=dimensional_params.forcing_config.oil_heating,
            )
        case _:
            raise NotImplementedError
