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
from dataclasses import field

from celestine.params.convection import NoBrineConvection
from celestine.params.initial_conditions import (
    BRW09InitialConditions,
    UniformInitialConditions,
)
from .params import (
    Config,
    NumericalParams,
)
from .convert import (
    get_dimensionless_brine_convection_params,
    get_dimensionless_bubble_params,
    get_dimensionless_forcing_config,
    get_dimensionless_initial_conditions_config,
    get_dimensionless_physical_params,
    calculate_timescale_in_days,
    calculate_velocity_scale_in_m_day,
    Scales,
)


@serde(type_check=coerce)
class DimensionalWaterParams:
    liquid_density: float = 1028  # kg/m3
    ocean_salinity: float = 34  # g/kg
    eutectic_salinity: float = 270  # g/kg
    eutectic_temperature: float = -21.1  # deg Celsius
    ocean_temperature: float = -0.81  # deg Celsius
    latent_heat: float = 334e3  # latent heat of fusion for ice in J/kg
    specific_heat_capacity: float = 4184  # ice and water assumed equal in J/kg degC
    # Option to average the conductivity term.
    phase_average_conductivity: bool = False
    liquid_thermal_conductivity: float = 0.54  # water thermal conductivity in W/m deg C
    solid_thermal_conductivity: float = 2.22  # ice thermal conductivity in W/m deg C

    salt_diffusivity: float = 0  # molecular diffusivity of salt in water in m2/s

    # calculated from moreau et al 2014 value of kinematic viscosity for sewater 2.7e-6
    # dynamic liquid_viscosity = 2.7e-6 * liquid_density
    liquid_viscosity: float = 2.78e-3  # dynamic liquid viscosity in Pa.s

    @property
    def salinity_difference(self):
        r"""calculate difference between eutectic salinity and typical ocean salinity

        .. math:: \Delta S = S_E - S_i

        """
        return self.eutectic_salinity - self.ocean_salinity

    @property
    def ocean_freezing_temperature(self):
        """calculate salinity dependent freezing temperature using liquidus for typical
        ocean salinity

        .. math:: T_i = T_L(S_i) = T_E S_i / S_E

        """
        return self.eutectic_temperature * self.ocean_salinity / self.eutectic_salinity

    @property
    def temperature_difference(self):
        r"""calculate

        .. math:: \Delta T = T_i - T_E

        """
        return self.ocean_freezing_temperature - self.eutectic_temperature

    @property
    def concentration_ratio(self):
        r"""Calculate concentration ratio as

        .. math:: \mathcal{C} = S_i / \Delta S

        """
        return self.ocean_salinity / self.salinity_difference

    @property
    def stefan_number(self):
        r"""calculate Stefan number

        .. math:: \text{St} = L / c_p \Delta T

        """
        return self.latent_heat / (
            self.temperature_difference * self.specific_heat_capacity
        )

    @property
    def thermal_diffusivity(self):
        r"""Return thermal diffusivity in m2/s

        .. math:: \kappa = \frac{k}{\rho_l c_p}

        """
        return self.liquid_thermal_conductivity / (
            self.liquid_density * self.specific_heat_capacity
        )

    @property
    def conductivity_ratio(self):
        r"""Calculate the ratio of solid to liquid thermal conductivity

        .. math:: \lambda = \frac{k_s}{k_l}

        """
        return self.solid_thermal_conductivity / self.liquid_thermal_conductivity

    @property
    def lewis_salt(self):
        r"""Calculate the lewis number for salt, return np.inf if there is no salt
        diffusion.

        .. math:: \text{Le}_S = \kappa / D_s

        """
        if self.salt_diffusivity == 0:
            return np.inf

        return self.thermal_diffusivity / self.salt_diffusivity


@serde(type_check=coerce)
class DimensionalGasParams:
    gas_density: float = 1  # kg/m3
    saturation_concentration: float = 1e-5  # kg(gas)/kg(liquid)
    ocean_saturation_state: float = 1.0  # fraction of saturation in ocean
    gas_diffusivity: float = 0  # molecular diffusivity of gas in water in m2/s
    # Option to change tolerable super saturation in brines
    tolerable_super_saturation_fraction: float = 1

    # timescale of nucleation to set damkohler number (in seconds)
    nucleation_timescale: float = 6869075


@serde(type_check=coerce)
class DimensionalBaseBubbleParams:
    pore_radius: float = 1e-3  # pore throat size scale in m
    pore_throat_scaling: float = 1 / 2
    porosity_threshold: bool = False
    porosity_threshold_value: float = 0.024


@serde(type_check=coerce)
class DimensionalMonoBubbleParams(DimensionalBaseBubbleParams):
    bubble_radius: float = 1e-3  # bubble radius in m

    @property
    def bubble_radius_scaled(self):
        r"""calculate the bubble radius divided by the pore scale

        .. math:: \Lambda = R_B / R_0

        """
        return self.bubble_radius / self.pore_radius


@serde(type_check=coerce)
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


@serde(type_check=coerce)
class DimensionalRJW14Params:
    couple_bubble_to_horizontal_flow: bool = False
    couple_bubble_to_vertical_flow: bool = False

    # Rees Jones and Worster 2014
    Rayleigh_critical: float = 2.9
    convection_strength: float = 0.13
    haline_contraction_coefficient: float = 7.5e-4
    reference_permeability: float = 1e-8


@serde(type_check=coerce)
class DimensionalSummerInitialcConditions:
    # Parameters for summer initial conditions
    initial_summer_ice_depth: float = 1  # in m
    initial_summer_ocean_temperature: float = -2  # in deg C
    initial_summer_ice_temperature: float = -4  # in deg C


@serde(type_check=coerce)
class DimensionalYearlyForcing:
    # These are the parameters for the sinusoidal temperature cycle in non dimensional
    # units
    offset: float = -1.0
    amplitude: float = 0.75
    period: float = 4.0


@serde(type_check=coerce)
class DimensionalRadForcing:
    # Short wave forcing parameters
    SW_internal_heating: bool = False
    SW_forcing_choice: str = "constant"
    constant_SW_irradiance: float = 280  # W/m2
    SW_radiation_model_choice: str = "1L"  # specify oilrad model to use
    constant_oil_mass_ratio: float = 0  # ng/g
    SW_scattering_ice_type: str = "FYI"

    # surface energy balance forcing parameters
    surface_energy_balance_forcing: bool = True


@serde(type_check=coerce)
class DimensionalConstantForcing:
    # Forcing configuration parameters
    constant_top_temperature: float = -30.32


@serde(type_check=coerce)
class DimensionalBRW09Forcing:
    Barrow_top_temperature_data_choice: str = "air"
    Barrow_initial_bulk_gas_in_ice: float = 1 / 5


@serde(type_check=coerce)
class DimensionalParams:
    """Contains all dimensional parameters needed to calculate non dimensional numbers.

    To see the units each input should have look at the comment next to the default
    value.
    """

    name: str
    total_time_in_days: float = 365  # days
    savefreq_in_days: float = 1  # save data after this amount of time in days
    lengthscale: float = 1  # domain height in m
    frame_velocity_dimensional: float = 0  # velocity of frame in m/day
    gravity: float = 9.81  # m/s2

    water_params: DimensionalWaterParams = field(default_factory=DimensionalWaterParams)
    gas_params: DimensionalGasParams = field(default_factory=DimensionalGasParams)
    bubble_params: DimensionalMonoBubbleParams | DimensionalPowerLawBubbleParams = (
        field(default_factory=DimensionalMonoBubbleParams)
    )
    brine_convection_params: DimensionalRJW14Params | NoBrineConvection = field(
        default_factory=DimensionalRJW14Params
    )
    forcing_config: DimensionalRadForcing | DimensionalBRW09Forcing | DimensionalConstantForcing | DimensionalYearlyForcing = field(
        default_factory=DimensionalBRW09Forcing
    )
    initial_conditions_config: DimensionalSummerInitialcConditions | UniformInitialConditions | BRW09InitialConditions = field(
        default_factory=BRW09InitialConditions
    )
    numerical_params: NumericalParams = field(default_factory=NumericalParams)

    @property
    def damkohler_number(self):
        r"""Return damkohler number as ratio of thermal timescale to nucleation
        timescale
        """
        return (
            (self.lengthscale**2) / self.water_params.thermal_diffusivity
        ) / self.gas_params.nucleation_timescale

    @property
    def total_time(self):
        """calculate the total time in non dimensional units for the simulation"""
        timescale = calculate_timescale_in_days(
            self.lengthscale, self.water_params.thermal_diffusivity
        )
        return self.total_time_in_days / timescale

    @property
    def savefreq(self):
        """calculate the save frequency in non dimensional time"""
        timescale = calculate_timescale_in_days(
            self.lengthscale, self.water_params.thermal_diffusivity
        )
        return self.savefreq_in_days / timescale

    @property
    def frame_velocity(self):
        """calculate the frame velocity in non dimensional units"""
        velocity_scale = calculate_velocity_scale_in_m_day(
            self.lengthscale, self.water_params.thermal_diffusivity
        )
        return self.frame_velocity_dimensional / velocity_scale

    @property
    def B(self):
        r"""calculate the non dimensional scale for buoyant rise of gas bubbles as

        .. math:: \mathcal{B} = \frac{\rho_l g R_0^2 h}{3 \mu \kappa}

        """
        stokes_velocity = (
            self.water_params.liquid_density
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
        return (
            self.water_params.liquid_density
            * self.gravity
            * self.brine_convection_params.haline_contraction_coefficient
            * self.water_params.salinity_difference
            * self.lengthscale
            * self.brine_convection_params.reference_permeability
            / (
                self.water_params.thermal_diffusivity
                * self.water_params.liquid_viscosity
            )
        )

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

    def get_config(self):
        """Return a Config object for the simulation.

        physical parameters and Darcy law parameters are calculated from the dimensional
        input. You can modify the numerical parameters and boundary conditions and
        forcing provided for the simulation."""
        physical_params = get_dimensionless_physical_params(self)
        initial_conditions_config = get_dimensionless_initial_conditions_config(self)
        brine_convection_params = get_dimensionless_brine_convection_params(self)
        bubble_params = get_dimensionless_bubble_params(self)
        forcing_config = get_dimensionless_forcing_config(self)
        return Config(
            name=self.name,
            physical_params=physical_params,
            initial_conditions_config=initial_conditions_config,
            brine_convection_params=brine_convection_params,
            bubble_params=bubble_params,
            forcing_config=forcing_config,
            numerical_params=self.numerical_params,
            scales=self.get_scales(),
            total_time=self.total_time,
            savefreq=self.savefreq,
        )

    def get_scales(self):
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
            self.gas_params.saturation_concentration,
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
