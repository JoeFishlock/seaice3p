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
class DimensionalConstantSWForcing:
    SW_irradiance: float = 280  # W/m2
    SW_albedo: float = 0.7
    SW_penetration_fraction: float = 0.4


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalBackgroundOilHeating:
    oil_mass_ratio: float = 0  # ng/g
    ice_type: str = "FYI"


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalMobileOilHeating:
    ice_type: str = "FYI"


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
class DimensionalRadForcing:
    # Short wave forcing parameters
    SW_forcing: DimensionalSWForcing = DimensionalConstantSWForcing()
    oil_heating: DimensionalOilHeating = DimensionalBackgroundOilHeating()


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalConstantForcing:
    # Forcing configuration parameters
    constant_top_temperature: float = -30.32


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalBRW09Forcing:
    Barrow_top_temperature_data_choice: str = "air"
