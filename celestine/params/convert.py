from typing import TYPE_CHECKING
from serde import serde, coerce


if TYPE_CHECKING:
    from .dimensional import DimensionalParams

from .forcing import (
    ForcingConfig,
    ConstantForcing,
    BRW09Forcing,
    YearlyForcing,
    RadForcing,
)
from .initial_conditions import (
    InitialConditionsConfig,
    UniformInitialConditions,
    BRW09InitialConditions,
    SummerInitialConditions,
)
from .physical import DISEQPhysicalParams, EQMPhysicalParams


def get_dimensionless_forcing_config(
    dimensional_params: "DimensionalParams",
) -> ForcingConfig:
    match dimensional_params.temperature_forcing_choice:
        case "constant":
            top_temp = (
                dimensional_params.constant_top_temperature
                - dimensional_params.ocean_freezing_temperature
            ) / dimensional_params.temperature_difference
            return ConstantForcing(constant_top_temperature=top_temp)
        case "yearly":
            return YearlyForcing(
                offset=dimensional_params.offset,
                amplitude=dimensional_params.amplitude,
                period=dimensional_params.period,
            )
        case "barrow_2009":
            return BRW09Forcing(
                Barrow_top_temperature_data_choice=dimensional_params.Barrow_top_temperature_data_choice,
                Barrow_initial_bulk_gas_in_ice=dimensional_params.Barrow_initial_bulk_gas_in_ice,
            )
        case "rad":
            return RadForcing(
                surface_energy_balance_forcing=dimensional_params.surface_energy_balance_forcing,
                SW_internal_heating=dimensional_params.SW_internal_heating,
                SW_forcing_choice=dimensional_params.SW_forcing_choice,
                constant_SW_irradiance=dimensional_params.constant_SW_irradiance,
                SW_radiation_model_choice=dimensional_params.SW_radiation_model_choice,
                constant_oil_mass_ratio=dimensional_params.constant_oil_mass_ratio,
                SW_scattering_ice_type=dimensional_params.SW_scattering_ice_type,
            )
        case _:
            raise NotImplementedError


def get_dimensionless_initial_conditions_config(
    dimensional_params: "DimensionalParams",
) -> InitialConditionsConfig:
    scales = dimensional_params.get_scales()
    match dimensional_params.initial_conditions_choice:
        case "uniform":
            return UniformInitialConditions()
        case "barrow_2009":
            return BRW09InitialConditions()
        case "summer":
            return SummerInitialConditions(
                initial_summer_ice_depth=dimensional_params.initial_summer_ice_depth
                / dimensional_params.lengthscale,
                initial_summer_ocean_temperature=scales.convert_from_dimensional_temperature(
                    dimensional_params.initial_summer_ocean_temperature
                ),
                initial_summer_ice_temperature=scales.convert_from_dimensional_temperature(
                    dimensional_params.initial_summer_ice_temperature
                ),
            )
        case _:
            raise NotImplementedError


def get_dimensionless_physical_params(dimensional_params: "DimensionalParams"):
    """return a PhysicalParams object"""
    match dimensional_params.model:
        case "EQM":
            return EQMPhysicalParams(
                expansion_coefficient=dimensional_params.expansion_coefficient,
                concentration_ratio=dimensional_params.concentration_ratio,
                stefan_number=dimensional_params.stefan_number,
                lewis_salt=dimensional_params.lewis_salt,
                lewis_gas=dimensional_params.lewis_gas,
                frame_velocity=dimensional_params.frame_velocity,
                phase_average_conductivity=dimensional_params.phase_average_conductivity,
                conductivity_ratio=dimensional_params.conductivity_ratio,
                tolerable_super_saturation_fraction=dimensional_params.tolerable_super_saturation_fraction,
            )
        case "DISEQ":
            return DISEQPhysicalParams(
                expansion_coefficient=dimensional_params.expansion_coefficient,
                concentration_ratio=dimensional_params.concentration_ratio,
                stefan_number=dimensional_params.stefan_number,
                lewis_salt=dimensional_params.lewis_salt,
                lewis_gas=dimensional_params.lewis_gas,
                frame_velocity=dimensional_params.frame_velocity,
                phase_average_conductivity=dimensional_params.phase_average_conductivity,
                conductivity_ratio=dimensional_params.conductivity_ratio,
                tolerable_super_saturation_fraction=dimensional_params.tolerable_super_saturation_fraction,
                damkohler_number=dimensional_params.damkohler_number,
            )
        case _:
            raise NotImplementedError


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


@serde(type_check=coerce)
class Scales:
    lengthscale: float  # domain height in m
    thermal_diffusivity: float  # m2/s
    liquid_thermal_conductivity: float  # W/m deg C
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

    def convert_from_dimensional_heating(self, dimensional_heating):
        """convert from heating rate in W/m3 to dimensionless units"""
        return (
            dimensional_heating
            * self.lengthscale**2
            / (self.liquid_thermal_conductivity * self.temperature_difference)
        )

    def convert_from_dimensional_heat_flux(self, dimensional_heat_flux):
        """convert from heat flux in W/m2 to dimensionless units"""
        return (
            dimensional_heat_flux
            * self.lengthscale
            / (self.liquid_thermal_conductivity * self.temperature_difference)
        )
