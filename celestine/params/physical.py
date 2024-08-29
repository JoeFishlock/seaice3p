import numpy as np
from serde import serde, coerce

from .dimensional import (
    DimensionalParams,
    DimensionalEQMGasParams,
    DimensionalDISEQGasParams,
)


@serde(type_check=coerce)
class BasePhysicalParams:
    """Not to be used directly but provides the common parameters for physical params
    objects
    """

    expansion_coefficient: float = 0.029
    concentration_ratio: float = 0.17
    stefan_number: float = 4.2
    lewis_salt: float = np.inf
    lewis_gas: float = np.inf
    frame_velocity: float = 0

    # Option to average the conductivity term.
    phase_average_conductivity: bool = False
    conductivity_ratio: float = 4.11

    # Option to change tolerable supersaturation
    tolerable_super_saturation_fraction: float = 1


@serde(type_check=coerce)
class EQMPhysicalParams(BasePhysicalParams):
    """non dimensional numbers for the mushy layer"""


@serde(type_check=coerce)
class DISEQPhysicalParams(BasePhysicalParams):
    """non dimensional numbers for the mushy layer"""

    # only used in DISEQ model
    damkohler_number: float = 1


PhysicalParams = EQMPhysicalParams | DISEQPhysicalParams


def get_dimensionless_physical_params(
    dimensional_params: DimensionalParams,
) -> PhysicalParams:
    """return a PhysicalParams object"""
    match dimensional_params.gas_params:
        case DimensionalEQMGasParams():
            return EQMPhysicalParams(
                expansion_coefficient=dimensional_params.expansion_coefficient,
                concentration_ratio=dimensional_params.water_params.concentration_ratio,
                stefan_number=dimensional_params.water_params.stefan_number,
                lewis_salt=dimensional_params.water_params.lewis_salt,
                lewis_gas=dimensional_params.lewis_gas,
                frame_velocity=dimensional_params.frame_velocity,
                phase_average_conductivity=dimensional_params.water_params.phase_average_conductivity,
                conductivity_ratio=dimensional_params.water_params.conductivity_ratio,
                tolerable_super_saturation_fraction=dimensional_params.gas_params.tolerable_super_saturation_fraction,
            )
        case DimensionalDISEQGasParams():
            return DISEQPhysicalParams(
                expansion_coefficient=dimensional_params.expansion_coefficient,
                concentration_ratio=dimensional_params.water_params.concentration_ratio,
                stefan_number=dimensional_params.water_params.stefan_number,
                lewis_salt=dimensional_params.water_params.lewis_salt,
                lewis_gas=dimensional_params.lewis_gas,
                frame_velocity=dimensional_params.frame_velocity,
                phase_average_conductivity=dimensional_params.water_params.phase_average_conductivity,
                conductivity_ratio=dimensional_params.water_params.conductivity_ratio,
                tolerable_super_saturation_fraction=dimensional_params.gas_params.tolerable_super_saturation_fraction,
                damkohler_number=dimensional_params.damkohler_number,
            )
        case _:
            raise NotImplementedError
