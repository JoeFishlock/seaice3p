"""Dimensional parameters required to run a simulation and convert output
to dimensional variables.

The DimensionalParams class contains all the dimensional parameters needed to produce
a simulation configuration.

The Scales class contains all the dimensional parameters required to convert simulation
output between physical and non-dimensional variables.
"""

from yaml import safe_load, dump
from dataclasses import dataclass, asdict
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
    timescale_in_days = calculate_timescale_in_days(lengthscale, thermal_diffusivity)
    return lengthscale / timescale_in_days


@dataclass
class DimensionalParams:
    name: str
    total_time_in_days: float = 365  # days
    savefreq_in_days: float = 1  # save data after this amount of time in days
    data_path: str = "data/"

    lengthscale: float = 1  # domain height in m
    liquid_density: float = 1028  # kg/m3
    gas_density: float = 1  # kg/m3
    saturation_concentration: float = 1e-5  # kg(gas)/kg(liquid)
    ocean_salinity: float = 34  # g/kg
    eutectic_salinity: float = 270  # g/kg
    eutectic_temperature: float = -21.1  # deg Celsius
    latent_heat: float = 334e3  # latent heat of fusion for ice in J/kg
    specific_heat_capacity: float = 4184  # ice and water assumed equal in J/kg degC
    thermal_conductivity: float = 0.598  # ice and water assumed equal in W/m degC
    salt_diffusivity: float = 0  # molecular diffusivity of salt in water in m2/s
    gas_diffusivity: float = 0  # molecular diffusivity of gas in water in m2/s
    frame_velocity_dimensional: float = 0  # velocity of frame in m/day

    gravity: float = 9.81  # m/s2
    liquid_viscosity: float = 8.9e-4  # Pa.s = N.s/m2 = kg/m.s
    bubble_radius: float = 1e-3  # bubble radius in m
    pore_radius: float = 1e-3  # pore throat size scale in m
    pore_throat_scaling: float = 1 / 2
    drag_exponent: float = 6.0
    liquid_velocity_dimensional: float = 0.0  # liquid darcy velocity in m/day

    @property
    def expansion_coefficient(self):
        return self.liquid_density * self.saturation_concentration / self.gas_density

    @property
    def salinity_difference(self):
        return self.eutectic_salinity - self.ocean_salinity

    @property
    def ocean_freezing_temperature(self):
        return self.eutectic_temperature * self.ocean_salinity / self.eutectic_salinity

    @property
    def temperature_difference(self):
        return self.ocean_freezing_temperature - self.eutectic_temperature

    @property
    def concentration_ratio(self):
        return self.ocean_salinity / self.salinity_difference

    @property
    def stefan_number(self):
        return self.latent_heat / (
            self.temperature_difference * self.specific_heat_capacity
        )

    @property
    def thermal_diffusivity(self):
        r"""Return thermal diffusivity in m2/s

        .. math:: \kappa = \frac{k}{\rho_l c_p}

        """
        return self.thermal_conductivity / (
            self.liquid_density * self.specific_heat_capacity
        )

    @property
    def lewis_salt(self):
        if self.salt_diffusivity == 0:
            return np.inf

        return self.thermal_diffusivity / self.salt_diffusivity

    @property
    def lewis_gas(self):
        if self.gas_diffusivity == 0:
            return np.inf

        return self.thermal_diffusivity / self.gas_diffusivity

    @property
    def total_time(self):
        timescale = calculate_timescale_in_days(
            self.lengthscale, self.thermal_diffusivity
        )
        return self.total_time_in_days / timescale

    @property
    def savefreq(self):
        timescale = calculate_timescale_in_days(
            self.lengthscale, self.thermal_diffusivity
        )
        return self.savefreq_in_days / timescale

    @property
    def frame_velocity(self):
        velocity_scale = calculate_velocity_scale_in_m_day(
            self.lengthscale, self.thermal_diffusivity
        )
        return self.frame_velocity_dimensional / velocity_scale

    @property
    def B(self):
        stokes_velocity = (
            self.liquid_density
            * self.gravity
            * self.bubble_radius**2
            / (3 * self.liquid_viscosity)
        )
        velocity_scale_in_m_per_second = self.thermal_diffusivity / self.lengthscale
        return stokes_velocity / velocity_scale_in_m_per_second

    @property
    def bubble_radius_scaled(self):
        return self.bubble_radius / self.pore_radius

    @property
    def liquid_velocity(self):
        velocity_scale_in_m_per_day = calculate_velocity_scale_in_m_day(
            self.lengthscale, self.thermal_diffusivity
        )
        return self.liquid_velocity_dimensional / velocity_scale_in_m_per_day

    def get_physical_params(self):
        return PhysicalParams(
            expansion_coefficient=self.expansion_coefficient,
            concentration_ratio=self.concentration_ratio,
            stefan_number=self.stefan_number,
            lewis_salt=self.lewis_salt,
            lewis_gas=self.lewis_gas,
            frame_velocity=self.frame_velocity,
        )

    def get_darcy_law_params(self):
        return DarcyLawParams(
            B=self.B,
            bubble_radius_scaled=self.bubble_radius_scaled,
            pore_throat_scaling=self.pore_throat_scaling,
            drag_exponent=self.drag_exponent,
            liquid_velocity=self.liquid_velocity,
        )

    def get_config(
        self,
        boundary_conditions_config: BoundaryConditionsConfig = BoundaryConditionsConfig(),
        forcing_config: ForcingConfig = ForcingConfig(),
        numerical_params: NumericalParams = NumericalParams(),
    ):
        physical_params = self.get_physical_params()
        darcy_law_params = self.get_darcy_law_params()
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
            data_path=self.data_path,
        )

    def get_scales(self):
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

    def save(self):
        with open(f"{self.data_path}{self.name}_dimensional.yml", "w") as outfile:
            dump(asdict(self), outfile)

    @classmethod
    def load(cls, path):
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
