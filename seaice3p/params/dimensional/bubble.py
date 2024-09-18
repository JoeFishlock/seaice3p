from dataclasses import dataclass
from serde import serde, coerce


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalBaseBubbleParams:
    pore_radius: float = 1e-3  # pore throat size scale in m
    pore_throat_scaling: float = 1 / 2
    porosity_threshold: bool = False
    porosity_threshold_value: float = 0.024
    escape_ice_surface: bool = True


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalMonoBubbleParams(DimensionalBaseBubbleParams):
    bubble_radius: float = 1e-3  # bubble radius in m

    @property
    def bubble_radius_scaled(self):
        r"""calculate the bubble radius divided by the pore scale

        .. math:: \Lambda = R_B / R_0

        """
        return self.bubble_radius / self.pore_radius


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalPowerLawBubbleParams(DimensionalBaseBubbleParams):
    bubble_distribution_power: float = 1.5
    minimum_bubble_radius: float = 1e-6
    maximum_bubble_radius: float = 1e-3

    @property
    def minimum_bubble_radius_scaled(self):
        r"""calculate the bubble radius divided by the pore scale

        .. math:: \Lambda = R_B / R_0

        """
        return self.minimum_bubble_radius / self.pore_radius

    @property
    def maximum_bubble_radius_scaled(self):
        r"""calculate the bubble radius divided by the pore scale

        .. math:: \Lambda = R_B / R_0

        """
        return self.maximum_bubble_radius / self.pore_radius
