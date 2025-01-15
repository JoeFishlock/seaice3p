from dataclasses import dataclass, field
from numpy.typing import NDArray
from serde import serde, coerce
from typing import Optional
from pathlib import Path
import oilrad as oi


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalYearlyForcing:
    # These are the parameters for the sinusoidal temperature cycle in non dimensional
    # units
    offset: float = -1.0
    amplitude: float = 0.75
    period: float = 4.0


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalConstantSWForcing:
    SW_irradiance: float = 280  # W/m2
    ice_scattering_coefficient: float = 1.71  # 1/m
    absorption_enhancement_factor: float = 2
    snow_spectral_albedos: NDArray = field(
        default_factory=lambda: oi.SNOW_ALBEDOS["light2022"]
    )
    snow_extinction_coefficients: NDArray = field(
        default_factory=lambda: oi.SNOW_EXTINCTION_COEFFICIENTS["lebrun2023"]
    )
    SSL_depth: float = 0.04  # m
    SSL_spectral_albedos: NDArray = field(
        default_factory=lambda: oi.SSL_ALBEDOS["light2022"]
    )
    SSL_extinction_coefficients: NDArray = field(
        default_factory=lambda: oi.SSL_EXTINCTION_COEFFICIENTS["perovich1990"]
    )


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalBackgroundOilHeating:
    oil_mass_ratio: float = 0  # ng/g
    median_oil_droplet_radius: float = 0.5  # microns


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalMobileOilHeating:
    pass


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalNoHeating:
    pass


DimensionalOilHeating = (
    DimensionalBackgroundOilHeating | DimensionalMobileOilHeating | DimensionalNoHeating
)
DimensionalSWForcing = DimensionalConstantSWForcing


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalConstantLWForcing:
    LW_irradiance: float = 260  # W/m2
    ice_emissitivty: float = 0.99
    water_emissivity: float = 0.97


DimensionalLWForcing = DimensionalConstantLWForcing


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalConstantTurbulentFlux:
    """Parameters for calculating the turbulent surface sensible and latent heat
    fluxes

    NOTE: If you are running a simulation with ERA5 reanalysis forcing you must set
    the ref_height=2m as this is the appropriate value for the atmospheric reanalysis
    quantities
    """

    ref_height: float = 10  # m
    windspeed: float = 5  # m/s
    air_temp: float = 0  # deg C
    specific_humidity: float = 3.6e-3  # kg water / kg air
    atm_pressure: float = 101.325  # KPa

    air_density: float = 1.275  # kg/m3
    air_heat_capacity: float = 1005  # J/kg K
    air_latent_heat_of_vaporisation: float = 2.501e6  # J/kg


DimensionalTurbulentFlux = DimensionalConstantTurbulentFlux


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalRadForcing:
    # Short wave forcing parameters
    SW_forcing: DimensionalSWForcing = DimensionalConstantSWForcing()
    LW_forcing: DimensionalLWForcing = DimensionalConstantLWForcing()
    turbulent_flux: DimensionalTurbulentFlux = DimensionalConstantTurbulentFlux()
    oil_heating: DimensionalOilHeating = DimensionalBackgroundOilHeating()


@serde(type_check=coerce)
class DimensionalERA5Forcing:
    """read ERA5 data from netCDF file located at data_path.

    Simulation will take atmospheric forcings from the start date specified in the
    format YYYY-MM-DD
    """

    data_path: Path
    start_date: str  # YYYY-MM-DD
    use_snow_data: bool = False
    SW_forcing: DimensionalSWForcing = DimensionalConstantSWForcing()
    LW_forcing: DimensionalLWForcing = DimensionalConstantLWForcing()
    turbulent_flux: DimensionalTurbulentFlux = DimensionalConstantTurbulentFlux()
    oil_heating: DimensionalOilHeating = DimensionalBackgroundOilHeating()


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalConstantForcing:
    # Forcing configuration parameters
    constant_top_temperature: float = -30.32


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalRobinForcing:
    """This forcing imposes a Robin boundary condition of the form
    surface_heat_flux=heat_transfer_coefficient * (restoring_temp - surface_temp)
    """

    heat_transfer_coefficient: float = 6.3  # W/m2K
    restoring_temperature: float = -30  # deg C


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalBRW09Forcing:
    Barrow_top_temperature_data_choice: str = "air"
