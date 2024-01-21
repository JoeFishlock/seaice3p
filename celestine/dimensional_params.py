"""Dimensional parameters required to run a simulation and convert output
to dimensional variables.

The DimensionalParams class contains all the dimensional parameters needed to produce
a simulation configuration.

The Scales class contains all the dimensional parameters required to convert simulation
output between physical and non-dimensional variables.
"""

from yaml import safe_load, dump
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from celestine.params import (
    Config,
    PhysicalParams,
    BoundaryConditionsConfig,
    ForcingConfig,
    NumericalParams,
    DarcyLawParams,
)

SECONDS_TO_DAYS = 1 / (60 * 60 * 24)


def calculate_timescale_in_days(lengthscale, thermal_diffusivity):
    """calculate timescale given domain height and thermal diffusivity.

    :param lengthscale: domain height in m
    :type lengthscale: float
    :param thermal_diffusivity: thermal diffusivity in m2/s
    :type thermal_diffusivity: float
    :return: timescale in days
    """
    return SECONDS_TO_DAYS * lengthscale**2 / thermal_diffusivity


def calculate_velocity_scale_in_m_day(lengthscale, thermal_diffusivity):
    """calculate the velocity scale given domain height and thermal diffusivity

    :param lengthscale: domain height in m
    :type lengthscale: float
    :param thermal_diffusivity: thermal diffusivity in m2/s
    :type thermal_diffusivity: float
    :return: velocity scale in m/day
    """
    timescale_in_days = calculate_timescale_in_days(lengthscale, thermal_diffusivity)
    return lengthscale / timescale_in_days


@dataclass
class DimensionalParams:
    """Contains all dimensional parameters needed to calculate non dimensional numbers.

    To see the units each input should have look at the comment next to the default
    value.
    """

    name: str
    total_time_in_days: float = 365  # days
    savefreq_in_days: float = 1  # save data after this amount of time in days

    lengthscale: float = 1  # domain height in m
    liquid_density: float = 1028  # kg/m3
    gas_density: float = 1  # kg/m3
    saturation_concentration: float = 1e-5  # kg(gas)/kg(liquid)
    ocean_salinity: float = 34  # g/kg
    eutectic_salinity: float = 270  # g/kg
    eutectic_temperature: float = -21.1  # deg Celsius
    latent_heat: float = 334e3  # latent heat of fusion for ice in J/kg
    specific_heat_capacity: float = 4184  # ice and water assumed equal in J/kg degC

    # Option to average the conductivity term.
    phase_average_conductivity: bool = False
    liquid_thermal_conductivity: float = 0.54  # water thermal conductivity in W/m deg C
    solid_thermal_conductivity: float = 2.22  # ice thermal conductivity in W/m deg C

    salt_diffusivity: float = 0  # molecular diffusivity of salt in water in m2/s
    gas_diffusivity: float = 0  # molecular diffusivity of gas in water in m2/s
    frame_velocity_dimensional: float = 0  # velocity of frame in m/day

    gravity: float = 9.81  # m/s2

    # calculated from moreau et al 2014 value of kinematic viscosity for sewater 2.7e-6
    # dynamic liquid_viscosity = 2.7e-6 * liquid_density
    liquid_viscosity: float = 2.78e-3  # dynamic liquid viscosity in Pa.s

    bubble_radius: float = 1e-3  # bubble radius in m
    pore_radius: float = 1e-3  # pore throat size scale in m
    pore_throat_scaling: float = 1 / 2
    drag_exponent: float = 6.0

    bubble_size_distribution_type: str = "mono"
    wall_drag_law_choice: str = "power"
    bubble_distribution_power: float = 1.5
    minimum_bubble_radius: float = 1e-6
    maximum_bubble_radius: float = 1e-3

    porosity_threshold: bool = False
    porosity_threshold_value: float = 0.024

    brine_convection_parameterisation: bool = False
    couple_bubble_to_horizontal_flow: bool = True
    couple_bubble_to_vertical_flow: bool = True

    # Rees Jones and Worster 2014
    Rayleigh_critical: float = 40
    convection_strength: float = 0.03
    haline_contraction_coefficient: float = 7.5e-4
    reference_permeability: float = 1e-8

    # Boundary conditions in dimensional units
    initial_conditions_choice: str = "uniform"
    far_gas_sat: float = saturation_concentration
    far_temp: float = -0.81
    far_bulk_salinity: float = ocean_salinity

    # Forcing configuration parameters
    temperature_forcing_choice: str = "constant"
    constant_top_temperature: float = -30.32
    Barrow_top_temperature_data_choice: str = "air"
    Barrow_initial_bulk_gas_in_ice: float = 1 / 5
    # These are the parameters for the sinusoidal temperature cycle in non dimensional
    # units
    offset: float = -1.0
    amplitude: float = 0.75
    period: float = 4.0

    # Numerical Params
    I: int = 50
    timestep: float = 2e-4
    regularisation: float = 1e-6
    solver: str = "SCI"

    @property
    def expansion_coefficient(self):
        r"""calculate

        .. math:: \chi = \rho_l \xi_{\text{sat}} / \rho_g

        """
        return self.liquid_density * self.saturation_concentration / self.gas_density

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
    def lewis_salt(self):
        r"""Calculate the lewis number for salt, return np.inf if there is no salt
        diffusion.

        .. math:: \text{Le}_S = \kappa / D_s

        """
        if self.salt_diffusivity == 0:
            return np.inf

        return self.thermal_diffusivity / self.salt_diffusivity

    @property
    def lewis_gas(self):
        r"""Calculate the lewis number for dissolved gas, return np.inf if there is no
        dissolved gas diffusion.

        .. math:: \text{Le}_\xi = \kappa / D_\xi

        """
        if self.gas_diffusivity == 0:
            return np.inf

        return self.thermal_diffusivity / self.gas_diffusivity

    @property
    def total_time(self):
        """calculate the total time in non dimensional units for the simulation"""
        timescale = calculate_timescale_in_days(
            self.lengthscale, self.thermal_diffusivity
        )
        return self.total_time_in_days / timescale

    @property
    def savefreq(self):
        """calculate the save frequency in non dimensional time"""
        timescale = calculate_timescale_in_days(
            self.lengthscale, self.thermal_diffusivity
        )
        return self.savefreq_in_days / timescale

    @property
    def frame_velocity(self):
        """calculate the frame velocity in non dimensional units"""
        velocity_scale = calculate_velocity_scale_in_m_day(
            self.lengthscale, self.thermal_diffusivity
        )
        return self.frame_velocity_dimensional / velocity_scale

    @property
    def B(self):
        r"""calculate the non dimensional scale for buoyant rise of gas bubbles as

        .. math:: \mathcal{B} = \frac{\rho_l g R_0^2 h}{3 \mu \kappa}

        """
        stokes_velocity = (
            self.liquid_density
            * self.gravity
            * self.pore_radius**2
            / (3 * self.liquid_viscosity)
        )
        velocity_scale_in_m_per_second = self.thermal_diffusivity / self.lengthscale
        return stokes_velocity / velocity_scale_in_m_per_second

    @property
    def bubble_radius_scaled(self):
        r"""calculate the bubble radius divided by the pore scale

        .. math:: \Lambda = R_B / R_0

        """
        return self.bubble_radius / self.pore_radius

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

    @property
    def Rayleigh_salt(self):
        r"""Calculate the haline Rayleigh number as

        .. math:: \text{Ra}_S = \frac{\rho_l g \beta \Delta S H K_0}{\kappa \mu}

        """
        return (
            self.liquid_density
            * self.gravity
            * self.haline_contraction_coefficient
            * self.salinity_difference
            * self.lengthscale
            * self.reference_permeability
            / (self.thermal_diffusivity * self.liquid_viscosity)
        )

    @property
    def conductivity_ratio(self):
        r"""Calculate the ratio of solid to liquid thermal conductivity

        .. math:: \lambda = \frac{k_s}{k_l}

        """
        return self.solid_thermal_conductivity / self.liquid_thermal_conductivity

    def get_physical_params(self):
        """return a PhysicalParams object"""
        return PhysicalParams(
            expansion_coefficient=self.expansion_coefficient,
            concentration_ratio=self.concentration_ratio,
            stefan_number=self.stefan_number,
            lewis_salt=self.lewis_salt,
            lewis_gas=self.lewis_gas,
            frame_velocity=self.frame_velocity,
            phase_average_conductivity=self.phase_average_conductivity,
            conductivity_ratio=self.conductivity_ratio,
        )

    def get_darcy_law_params(self):
        """return a DarcyLawParams object"""
        return DarcyLawParams(
            B=self.B,
            bubble_radius_scaled=self.bubble_radius_scaled,
            pore_throat_scaling=self.pore_throat_scaling,
            drag_exponent=self.drag_exponent,
            bubble_size_distribution_type=self.bubble_size_distribution_type,
            wall_drag_law_choice=self.wall_drag_law_choice,
            bubble_distribution_power=self.bubble_distribution_power,
            minimum_bubble_radius_scaled=self.minimum_bubble_radius_scaled,
            maximum_bubble_radius_scaled=self.maximum_bubble_radius_scaled,
            porosity_threshold=self.porosity_threshold,
            porosity_threshold_value=self.porosity_threshold_value,
            brine_convection_parameterisation=self.brine_convection_parameterisation,
            Rayleigh_salt=self.Rayleigh_salt,
            Rayleigh_critical=self.Rayleigh_critical,
            convection_strength=self.convection_strength,
            couple_bubble_to_horizontal_flow=self.couple_bubble_to_horizontal_flow,
            couple_bubble_to_vertical_flow=self.couple_bubble_to_vertical_flow,
        )

    def get_boundary_conditions_config(self):
        return BoundaryConditionsConfig(
            initial_conditions_choice=self.initial_conditions_choice,
            far_gas_sat=self.far_gas_sat / self.saturation_concentration,
            far_temp=(self.far_temp - self.ocean_freezing_temperature)
            / self.temperature_difference,
            far_bulk_salinity=(self.far_bulk_salinity - self.ocean_salinity)
            / self.salinity_difference,
        )

    def get_forcing_config(self):
        return ForcingConfig(
            temperature_forcing_choice=self.temperature_forcing_choice,
            constant_top_temperature=(
                self.constant_top_temperature - self.ocean_freezing_temperature
            )
            / self.temperature_difference,
            offset=self.offset,
            amplitude=self.amplitude,
            period=self.period,
            Barrow_top_temperature_data_choice=self.Barrow_top_temperature_data_choice,
            Barrow_initial_bulk_gas_in_ice=self.Barrow_initial_bulk_gas_in_ice,
        )

    def get_numerical_params(self):
        return NumericalParams(
            I=self.I,
            timestep=self.timestep,
            regularisation=self.regularisation,
            solver=self.solver,
        )

    def get_config(self):
        """Return a Config object for the simulation.

        physical parameters and Darcy law parameters are calculated from the dimensional
        input. You can modify the numerical parameters and boundary conditions and
        forcing provided for the simulation."""
        physical_params = self.get_physical_params()
        darcy_law_params = self.get_darcy_law_params()
        boundary_conditions_config = self.get_boundary_conditions_config()
        forcing_config = self.get_forcing_config()
        numerical_params = self.get_numerical_params()
        return Config(
            name=self.name,
            physical_params=physical_params,
            boundary_conditions_config=boundary_conditions_config,
            darcy_law_params=darcy_law_params,
            forcing_config=forcing_config,
            numerical_params=numerical_params,
            scales=self.get_scales(),
            total_time=self.total_time,
            savefreq=self.savefreq,
        )

    def get_scales(self):
        """return a Scales object used for converting between dimensional and non
        dimensional variables."""
        return Scales(
            self.lengthscale,
            self.thermal_diffusivity,
            self.ocean_salinity,
            self.salinity_difference,
            self.ocean_freezing_temperature,
            self.temperature_difference,
            self.gas_density,
            self.saturation_concentration,
        )

    def save(self, directory: Path):
        """save this object to a yaml file in the specified directory.

        The name will be the name given with _dimensional appended to distinguish it
        from a saved non-dimensional configuration."""
        with open(directory / f"{self.name}_dimensional.yml", "w") as outfile:
            dump(asdict(self), outfile)

    @classmethod
    def load(cls, path):
        """load this object from a yaml configuration file."""
        with open(path, "r") as infile:
            dictionary = safe_load(infile)
        return cls(**dictionary)


@dataclass
class Scales:
    lengthscale: float  # domain height in m
    thermal_diffusivity: float  # m2/s
    ocean_salinity: float  # g/kg
    salinity_difference: float  # g/kg
    ocean_freezing_temperature: float  # deg C
    temperature_difference: float  # deg C
    gas_density: float  # kg/m3
    saturation_concentration: float  # kg(gas)/kg(liquid)

    @property
    def timescale_in_days(self):
        return calculate_timescale_in_days(self.lengthscale, self.thermal_diffusivity)

    @property
    def velocity_scale_in_m_per_day(self):
        return calculate_velocity_scale_in_m_day(
            self.lengthscale, self.thermal_diffusivity
        )

    def convert_from_dimensional_temperature(self, dimensional_temperature):
        """Non dimensionalise temperature in deg C"""
        return (
            dimensional_temperature - self.ocean_freezing_temperature
        ) / self.temperature_difference

    def convert_to_dimensional_temperature(self, temperature):
        """get temperature in deg C from non dimensional temperature"""
        return (
            self.temperature_difference * temperature + self.ocean_freezing_temperature
        )

    def convert_from_dimensional_grid(self, dimensional_grid):
        """Non dimensionalise domain depths in meters"""
        return dimensional_grid / self.lengthscale

    def convert_to_dimensional_grid(self, grid):
        """Get domain depths in meters from non dimensional values"""
        return self.lengthscale * grid

    def convert_from_dimensional_time(self, dimensional_time):
        """Non dimensionalise time in days"""
        return dimensional_time / self.timescale_in_days

    def convert_to_dimensional_time(self, time):
        """Convert non dimensional time into time in days since start of simulation"""
        return self.timescale_in_days * time

    def convert_from_dimensional_bulk_salinity(self, dimensional_bulk_salinity):
        """Non dimensionalise bulk salinity in g/kg"""
        return (
            dimensional_bulk_salinity - self.ocean_salinity
        ) / self.salinity_difference

    def convert_to_dimensional_bulk_salinity(self, bulk_salinity):
        """Convert non dimensional bulk salinity to g/kg"""
        return self.salinity_difference * bulk_salinity + self.ocean_salinity

    def convert_from_dimensional_bulk_gas(self, dimensional_bulk_gas):
        """Non dimensionalise bulk gas content in kg/m3"""
        return dimensional_bulk_gas / self.gas_density

    def convert_to_dimensional_bulk_gas(self, bulk_gas):
        """Convert dimensionless bulk gas content to kg/m3"""
        return self.gas_density * bulk_gas

    def convert_dimensional_bulk_air_to_argon_content(self, dimensional_bulk_gas):
        """Convert kg/m3 of air to micromole of Argon per Liter of ice"""
        mass_ratio_of_argon_in_air = 0.01288
        micromoles_of_argon_in_a_kilogram_of_argon = 1 / (3.9948e-8)
        liters_in_a_meter_cubed = 1e3
        return (
            dimensional_bulk_gas
            * mass_ratio_of_argon_in_air
            * micromoles_of_argon_in_a_kilogram_of_argon
            / liters_in_a_meter_cubed
        )

    def convert_from_dimensional_dissolved_gas(self, dimensional_dissolved_gas):
        """convert from dissolved gas in kg(gas)/kg(liquid) to dimensionless"""
        return dimensional_dissolved_gas / self.saturation_concentration

    def convert_to_dimensional_dissolved_gas(self, dissolved_gas):
        """convert from non dimensional dissolved gas to dimensional dissolved gas in
        kg(gas)/kg(liquid)"""
        return self.saturation_concentration * dissolved_gas
