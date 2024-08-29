from dataclasses import dataclass
from serde import serde, coerce


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
class DimensionalRadForcing:
    # Short wave forcing parameters
    SW_internal_heating: bool = False
    SW_forcing_choice: str = "constant"
    constant_SW_irradiance: float = 280  # W/m2
    SW_radiation_model_choice: str = "1L"  # specify oilrad model to use
    constant_oil_mass_ratio: float = 0  # ng/g
    SW_scattering_ice_type: str = "FYI"

    # surface energy balance forcing parameters
    surface_energy_balance_forcing: bool = True


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalConstantForcing:
    # Forcing configuration parameters
    constant_top_temperature: float = -30.32


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalBRW09Forcing:
    Barrow_top_temperature_data_choice: str = "air"
