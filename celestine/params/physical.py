import numpy as np
from serde import serde, coerce


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
