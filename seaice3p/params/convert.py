from dataclasses import dataclass
from serde import serde, coerce


SECONDS_TO_DAYS = 1 / (60 * 60 * 24)


@serde(type_check=coerce)
@dataclass(frozen=True)
class Scales:
    lengthscale: float  # domain height in m
    thermal_diffusivity: float  # m2/s
    liquid_thermal_conductivity: float  # W/m deg C
    ocean_salinity: float  # g/kg
    salinity_difference: float  # g/kg
    ocean_freezing_temperature: float  # deg C
    temperature_difference: float  # deg C
    gas_density: float  # kg/m3
    liquid_density: float  # kg/m3
    ice_density: float  # kg/m3
    saturation_concentration: float  # kg(gas)/kg(liquid)
    pore_radius: float  # m
    haline_contraction_coefficient: float  # 1/ppt

    @property
    def time_scale(self):
        """in days"""
        return SECONDS_TO_DAYS * self.lengthscale**2 / self.thermal_diffusivity

    @property
    def velocity_scale(self):
        """in m /day"""
        return self.lengthscale / self.time_scale

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
        return dimensional_time / self.time_scale

    def convert_to_dimensional_time(self, time):
        """Convert non dimensional time into time in days since start of simulation"""
        return self.time_scale * time

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
