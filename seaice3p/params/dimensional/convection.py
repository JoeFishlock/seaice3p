from dataclasses import dataclass
from serde import serde, coerce


@serde(type_check=coerce)
@dataclass(frozen=True)
class NoBrineConvection:
    """No brine convection"""


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalRJW14Params:
    couple_bubble_to_horizontal_flow: bool = False
    couple_bubble_to_vertical_flow: bool = False

    # Rees Jones and Worster 2014
    Rayleigh_critical: float = 2.9
    convection_strength: float = 0.13
    reference_permeability: float = 1e-8
