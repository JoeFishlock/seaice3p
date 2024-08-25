from serde import serde, coerce


@serde(type_check=coerce)
class UniformInitialConditions:
    """values for bottom (ocean) boundary"""


@serde(type_check=coerce)
class BRW09InitialConditions:
    """values for bottom (ocean) boundary"""


@serde(type_check=coerce)
class SummerInitialConditions:
    """values for bottom (ocean) boundary"""

    # Non dimensional parameters for summer initial conditions
    initial_summer_ice_depth: float = 0.5
    initial_summer_ocean_temperature: float = -0.05
    initial_summer_ice_temperature: float = -0.1


InitialConditionsConfig = (
    UniformInitialConditions | BRW09InitialConditions | SummerInitialConditions
)
