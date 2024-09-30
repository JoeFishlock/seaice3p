from dataclasses import dataclass
from serde import serde, coerce
from .dimensional import (
    DimensionalParams,
    DimensionalOilInitialConditions,
    UniformInitialConditions,
    BRW09InitialConditions,
    PreviousSimulation,
)


@serde(type_check=coerce)
@dataclass(frozen=True)
class OilInitialConditions:
    """values for bottom (ocean) boundary"""

    # Non dimensional parameters for summer initial conditions
    initial_ice_depth: float = 0.5
    initial_ocean_temperature: float = -0.05
    initial_ice_temperature: float = -0.1
    initial_oil_volume_fraction: float = 1e-7
    initial_ice_bulk_salinity: float = -0.1
    initial_oil_free_depth: float = 0


InitialConditionsConfig = (
    UniformInitialConditions
    | BRW09InitialConditions
    | OilInitialConditions
    | PreviousSimulation
)


def get_dimensionless_initial_conditions_config(
    dimensional_params: DimensionalParams,
) -> InitialConditionsConfig:
    scales = dimensional_params.scales
    match dimensional_params.initial_conditions_config:
        case UniformInitialConditions():
            return UniformInitialConditions()
        case BRW09InitialConditions():
            return BRW09InitialConditions(
                Barrow_initial_bulk_gas_in_ice=dimensional_params.initial_conditions_config.Barrow_initial_bulk_gas_in_ice
            )
        case DimensionalOilInitialConditions():
            return OilInitialConditions(
                initial_ice_depth=dimensional_params.initial_conditions_config.initial_ice_depth
                / dimensional_params.lengthscale,
                initial_ocean_temperature=scales.convert_from_dimensional_temperature(
                    dimensional_params.initial_conditions_config.initial_ocean_temperature
                ),
                initial_ice_temperature=scales.convert_from_dimensional_temperature(
                    dimensional_params.initial_conditions_config.initial_ice_temperature
                ),
                initial_oil_volume_fraction=dimensional_params.initial_conditions_config.initial_oil_volume_fraction,
                initial_ice_bulk_salinity=scales.convert_from_dimensional_bulk_salinity(
                    dimensional_params.initial_conditions_config.initial_ice_bulk_salinity
                ),
                initial_oil_free_depth=dimensional_params.initial_conditions_config.initial_oil_free_depth
                / dimensional_params.lengthscale,
            )
        case PreviousSimulation():
            return dimensional_params.initial_conditions_config
        case _:
            raise NotImplementedError
