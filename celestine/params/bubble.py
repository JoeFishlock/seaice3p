from serde import serde, coerce


@serde(type_check=coerce)
class BaseBubbleParams:
    """Not to be used directly but provides parameters for bubble model in sea ice
    common to other bubble parameter objects.
    """

    B: float = 100
    pore_throat_scaling: float = 0.46
    porosity_threshold: bool = False
    porosity_threshold_value: float = 0.024


@serde(type_check=coerce)
class MonoBubbleParams(BaseBubbleParams):
    """Parameters for population of identical spherical bubbles."""

    bubble_radius_scaled: float = 1.0


@serde(type_check=coerce)
class PowerLawBubbleParams(BaseBubbleParams):
    """Parameters for population of bubbles following a power law size distribution
    between a minimum and maximum radius.
    """

    bubble_distribution_power: float = 1.5
    minimum_bubble_radius_scaled: float = 1e-3
    maximum_bubble_radius_scaled: float = 1


BubbleParams = MonoBubbleParams | PowerLawBubbleParams
