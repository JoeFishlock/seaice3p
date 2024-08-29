from dataclasses import dataclass
from serde import serde, coerce
from .dimensional import (
    DimensionalParams,
    DimensionalSummerInitialConditions,
    UniformInitialConditions,
    BRW09InitialConditions,
)


@serde(type_check=coerce)
@dataclass(frozen=True)
class SummerInitialConditions:
    """values for bottom (ocean) boundary"""

    # Non dimensional parameters for summer initial conditions
    initial_summer_ice_depth: float = 0.5
    initial_summer_ocean_temperature: float = -0.05
    initial_summer_ice_temperature: float = -0.1


InitialConditionsConfig = (
    UniformInitialConditions | BRW09InitialConditions | SummerInitialConditions
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
        case DimensionalSummerInitialConditions():
            return SummerInitialConditions(
                initial_summer_ice_depth=dimensional_params.initial_conditions_config.initial_summer_ice_depth
                / dimensional_params.lengthscale,
                initial_summer_ocean_temperature=scales.convert_from_dimensional_temperature(
                    dimensional_params.initial_conditions_config.initial_summer_ocean_temperature
                ),
                initial_summer_ice_temperature=scales.convert_from_dimensional_temperature(
                    dimensional_params.initial_conditions_config.initial_summer_ice_temperature
                ),
            )
        case _:
            raise NotImplementedError
