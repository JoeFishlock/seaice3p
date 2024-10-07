from dataclasses import dataclass
from serde import serde, coerce


@serde(type_check=coerce)
@dataclass(frozen=True)
class _DimensionalGasParams:
    gas_density: float = 1  # kg/m3
    saturation_concentration: float = 1e-5  # kg(gas)/kg(liquid)
    ocean_saturation_state: float = 1.0  # fraction of saturation in ocean
    gas_diffusivity: float = 0  # molecular diffusivity of gas in water in m2/s
    # Option to change tolerable super saturation in brines
    tolerable_super_saturation_fraction: float = 1
    gas_viscosity: float = 0  # Pa s
    gas_bubble_eddy_diffusion: bool = False


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalEQMGasParams(_DimensionalGasParams):
    pass


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalDISEQGasParams(_DimensionalGasParams):
    # timescale of nucleation to set damkohler number (in seconds)
    nucleation_timescale: float = 6869075
