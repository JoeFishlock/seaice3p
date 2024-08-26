from serde import serde, coerce


@serde(type_check=coerce)
class RJW14Params:
    """Parameters for the RJW14 parameterisation of brine convection"""

    Rayleigh_salt: float = 44105
    Rayleigh_critical: float = 2.9
    convection_strength: float = 0.13
    couple_bubble_to_horizontal_flow: bool = False
    couple_bubble_to_vertical_flow: bool = False


@serde(type_check=coerce)
class NoBrineConvection:
    """No brine convection"""


BrineConvectionParams = RJW14Params | NoBrineConvection
