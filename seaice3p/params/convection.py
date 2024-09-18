from dataclasses import dataclass
from serde import serde, coerce
from .dimensional import DimensionalParams, DimensionalRJW14Params, NoBrineConvection


@serde(type_check=coerce)
@dataclass(frozen=True)
class RJW14Params:
    """Parameters for the RJW14 parameterisation of brine convection"""

    Rayleigh_salt: float = 44105
    Rayleigh_critical: float = 2.9
    convection_strength: float = 0.13
    couple_bubble_to_horizontal_flow: bool = False
    couple_bubble_to_vertical_flow: bool = False


BrineConvectionParams = RJW14Params | NoBrineConvection


def get_dimensionless_brine_convection_params(
    dimensional_params: DimensionalParams,
) -> BrineConvectionParams:
    match dimensional_params.brine_convection_params:
        case DimensionalRJW14Params():
            return RJW14Params(
                Rayleigh_salt=dimensional_params.Rayleigh_salt,
                Rayleigh_critical=dimensional_params.brine_convection_params.Rayleigh_critical,
                convection_strength=dimensional_params.brine_convection_params.convection_strength,
                couple_bubble_to_horizontal_flow=dimensional_params.brine_convection_params.couple_bubble_to_horizontal_flow,
                couple_bubble_to_vertical_flow=dimensional_params.brine_convection_params.couple_bubble_to_vertical_flow,
            )
        case NoBrineConvection():
            return NoBrineConvection()
