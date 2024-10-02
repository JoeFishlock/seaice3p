"""Dimensional parameters required to run a simulation and convert output
to dimensional variables.

The DimensionalParams class contains all the dimensional parameters needed to produce
a simulation configuration.

The Scales class contains all the dimensional parameters required to convert simulation
output between physical and non-dimensional variables.
"""

from pathlib import Path
import numpy as np
from serde import serde, coerce
from serde.yaml import from_yaml, to_yaml
from dataclasses import dataclass

from ..convert import (
    Scales,
)
from .water import DimensionalWaterParams
from .gas import DimensionalDISEQGasParams, DimensionalEQMGasParams
from .bubble import DimensionalMonoBubbleParams, DimensionalPowerLawBubbleParams
from .convection import NoBrineConvection, DimensionalRJW14Params
from .forcing import (
    DimensionalBRW09Forcing,
    DimensionalConstantForcing,
    DimensionalERA5Forcing,
    DimensionalRadForcing,
    DimensionalYearlyForcing,
    DimensionalRobinForcing,
)
from .ocean_forcing import (
    DimensionalBRW09OceanForcing,
    DimensionalFixedHeatFluxOceanForcing,
    DimensionalFixedTempOceanForcing,
)
from .initial_conditions import (
    DimensionalOilInitialConditions,
    BRW09InitialConditions,
    UniformInitialConditions,
    PreviousSimulation,
)
from .numerical import (
    NumericalParams,
)


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalParams:
    """Contains all dimensional parameters needed to calculate non dimensional numbers.

    To see the units each input should have look at the comment next to the default
    value.
    """

    name: str
    total_time_in_days: float
    savefreq_in_days: float
    lengthscale: float

    gas_params: DimensionalEQMGasParams | DimensionalDISEQGasParams
    bubble_params: DimensionalMonoBubbleParams | DimensionalPowerLawBubbleParams
    brine_convection_params: DimensionalRJW14Params | NoBrineConvection
    forcing_config: DimensionalRadForcing | DimensionalBRW09Forcing | DimensionalConstantForcing | DimensionalYearlyForcing | DimensionalRobinForcing | DimensionalERA5Forcing
    ocean_forcing_config: DimensionalBRW09OceanForcing | DimensionalFixedTempOceanForcing | DimensionalFixedHeatFluxOceanForcing
    initial_conditions_config: DimensionalOilInitialConditions | UniformInitialConditions | BRW09InitialConditions | PreviousSimulation

    water_params: DimensionalWaterParams = DimensionalWaterParams()
    numerical_params: NumericalParams = NumericalParams()
    frame_velocity_dimensional: float = 0  # velocity of frame in m/day
    gravity: float = 9.81  # m/s2

    @property
    def damkohler_number(self):
        r"""Return damkohler number as ratio of thermal timescale to nucleation
        timescale
        """
        if isinstance(self.gas_params, DimensionalEQMGasParams):
            return None

        return (
            (self.lengthscale**2) / self.water_params.thermal_diffusivity
        ) / self.gas_params.nucleation_timescale

    @property
    def total_time(self):
        """calculate the total time in non dimensional units for the simulation"""
        return self.total_time_in_days / self.scales.time_scale

    @property
    def savefreq(self):
        """calculate the save frequency in non dimensional time"""
        return self.savefreq_in_days / self.scales.time_scale

    @property
    def frame_velocity(self):
        """calculate the frame velocity in non dimensional units"""
        return self.frame_velocity_dimensional / self.scales.velocity_scale

    @property
    def B(self):
        r"""calculate the non dimensional scale for buoyant rise of gas bubbles as

        .. math:: \mathcal{B} = \frac{\rho_l g R_0^2 h}{3 \mu \kappa}

        """
        stokes_velocity = (
            (self.water_params.liquid_density - self.gas_params.gas_density)
            * self.gravity
            * self.bubble_params.pore_radius**2
            / (3 * self.water_params.liquid_viscosity)
        )
        velocity_scale_in_m_per_second = (
            self.water_params.thermal_diffusivity / self.lengthscale
        )
        return stokes_velocity / velocity_scale_in_m_per_second

    @property
    def Rayleigh_salt(self):
        r"""Calculate the haline Rayleigh number as

        .. math:: \text{Ra}_S = \frac{\rho_l g \beta \Delta S H K_0}{\kappa \mu}

        """
        match self.brine_convection_params:
            case DimensionalRJW14Params():
                return (
                    self.water_params.liquid_density
                    * self.gravity
                    * self.water_params.haline_contraction_coefficient
                    * self.water_params.salinity_difference
                    * self.lengthscale
                    * self.brine_convection_params.reference_permeability
                    / (
                        self.water_params.thermal_diffusivity
                        * self.water_params.liquid_viscosity
                    )
                )
            case NoBrineConvection():
                return None

    @property
    def expansion_coefficient(self):
        r"""calculate

        .. math:: \chi = \rho_l \xi_{\text{sat}} / \rho_g

        """
        return (
            self.water_params.liquid_density
            * self.gas_params.saturation_concentration
            / self.gas_params.gas_density
        )

    @property
    def lewis_gas(self):
        r"""Calculate the lewis number for dissolved gas, return np.inf if there is no
        dissolved gas diffusion.

        .. math:: \text{Le}_\xi = \kappa / D_\xi

        """
        if self.gas_params.gas_diffusivity == 0:
            return np.inf

        return self.water_params.thermal_diffusivity / self.gas_params.gas_diffusivity

    @property
    def scales(self):
        """return a Scales object used for converting between dimensional and non
        dimensional variables."""
        return Scales(
            self.lengthscale,
            self.water_params.thermal_diffusivity,
            self.water_params.liquid_thermal_conductivity,
            self.water_params.ocean_salinity,
            self.water_params.salinity_difference,
            self.water_params.ocean_freezing_temperature,
            self.water_params.temperature_difference,
            self.gas_params.gas_density,
            self.water_params.liquid_density,
            self.water_params.ice_density,
            self.gas_params.saturation_concentration,
            self.bubble_params.pore_radius,
            self.water_params.haline_contraction_coefficient,
        )

    def save(self, directory: Path):
        """save this object to a yaml file in the specified directory.

        The name will be the name given with _dimensional appended to distinguish it
        from a saved non-dimensional configuration."""
        with open(directory / f"{self.name}_dimensional.yml", "w") as outfile:
            outfile.write(to_yaml(self))

    @classmethod
    def load(cls, path):
        """load this object from a yaml configuration file."""
        with open(path, "r") as infile:
            yaml = infile.read()
        return from_yaml(cls, yaml)
