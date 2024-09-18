from dataclasses import dataclass
from serde import serde, coerce
from .dimensional import (
    DimensionalParams,
    DimensionalPowerLawBubbleParams,
    DimensionalMonoBubbleParams,
)


@serde(type_check=coerce)
@dataclass(frozen=True)
class BaseBubbleParams:
    """Not to be used directly but provides parameters for bubble model in sea ice
    common to other bubble parameter objects.
    """

    B: float = 100
    pore_throat_scaling: float = 0.46
    porosity_threshold: bool = False
    porosity_threshold_value: float = 0.024
    escape_ice_surface: bool = True


@serde(type_check=coerce)
@dataclass(frozen=True)
class MonoBubbleParams(BaseBubbleParams):
    """Parameters for population of identical spherical bubbles."""

    bubble_radius_scaled: float = 1.0


@serde(type_check=coerce)
@dataclass(frozen=True)
class PowerLawBubbleParams(BaseBubbleParams):
    """Parameters for population of bubbles following a power law size distribution
    between a minimum and maximum radius.
    """

    bubble_distribution_power: float = 1.5
    minimum_bubble_radius_scaled: float = 1e-3
    maximum_bubble_radius_scaled: float = 1


BubbleParams = MonoBubbleParams | PowerLawBubbleParams


def get_dimensionless_bubble_params(
    dimensional_params: DimensionalParams,
) -> BubbleParams:
    common_params = {
        "B": dimensional_params.B,
        "pore_throat_scaling": dimensional_params.bubble_params.pore_throat_scaling,
        "porosity_threshold": dimensional_params.bubble_params.porosity_threshold,
        "porosity_threshold_value": dimensional_params.bubble_params.porosity_threshold_value,
        "escape_ice_surface": dimensional_params.bubble_params.escape_ice_surface,
    }
    match dimensional_params.bubble_params:
        case DimensionalMonoBubbleParams():
            return MonoBubbleParams(
                **common_params,
                bubble_radius_scaled=dimensional_params.bubble_params.bubble_radius_scaled,
            )
        case DimensionalPowerLawBubbleParams():
            return PowerLawBubbleParams(
                **common_params,
                bubble_distribution_power=dimensional_params.bubble_params.bubble_distribution_power,
                minimum_bubble_radius_scaled=dimensional_params.bubble_params.minimum_bubble_radius_scaled,
                maximum_bubble_radius_scaled=dimensional_params.bubble_params.maximum_bubble_radius_scaled,
            )
        case _:
            raise NotImplementedError
