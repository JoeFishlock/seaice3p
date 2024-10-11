from dataclasses import dataclass
import numpy as np
from serde import serde, coerce


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalWaterParams:
    liquid_density: float = 1028  # kg/m3
    ice_density: float = 916  # kg/m3
    ocean_salinity: float = 34  # g/kg
    eutectic_salinity: float = 270  # g/kg
    eutectic_temperature: float = -21.1  # deg Celsius
    latent_heat: float = 334e3  # latent heat of fusion for ice in J/kg
    liquid_specific_heat_capacity: float = 4184  # J/kg degC
    solid_specific_heat_capacity: float = 2009  # J/kg degC
    liquid_thermal_conductivity: float = 0.54  # water thermal conductivity in W/m deg C
    solid_thermal_conductivity: float = 2.22  # ice thermal conductivity in W/m deg C
    snow_thermal_conductivity: float = 0.31  # snow thermal conductivity in W/m deg C

    eddy_diffusivity: float = 0

    salt_diffusivity: float = 0  # molecular diffusivity of salt in water in m2/s
    # used to calculate Rayleigh number for convection and density contraction in liquid equation of state
    haline_contraction_coefficient: float = 7.5e-4  # 1/ppt

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
            self.temperature_difference * self.liquid_specific_heat_capacity
        )

    @property
    def thermal_diffusivity(self):
        r"""Return thermal diffusivity in m2/s

        .. math:: \kappa = \frac{k}{\rho_l c_p}

        """
        return self.liquid_thermal_conductivity / (
            self.liquid_density * self.liquid_specific_heat_capacity
        )

    @property
    def conductivity_ratio(self):
        r"""Calculate the ratio of solid to liquid thermal conductivity

        .. math:: \lambda = \frac{k_s}{k_l}

        """
        return self.solid_thermal_conductivity / self.liquid_thermal_conductivity

    @property
    def specific_heat_ratio(self):
        r"""Calculate the ratio of solid to liquid specific heat capacities

        .. math:: \lambda = \frac{c_{p,s}}{c_{p,l}}

        """
        return self.solid_specific_heat_capacity / self.liquid_specific_heat_capacity

    @property
    def eddy_diffusivity_ratio(self):
        r"""Calculate the ratio of eddy diffusivity to thermal diffusivity in
        the liquid phase

        .. math:: \lambda = \frac{\kappa_\text{turbulent}}{\kappa_l}

        """
        return self.eddy_diffusivity / self.thermal_diffusivity

    @property
    def snow_conductivity_ratio(self):
        r"""Calculate the ratio of snow to liquid thermal conductivity

        .. math:: \lambda = \frac{k_{sn}}{k_l}

        """
        return self.snow_thermal_conductivity / self.liquid_thermal_conductivity

    @property
    def lewis_salt(self):
        r"""Calculate the lewis number for salt, return np.inf if there is no salt
        diffusion.

        .. math:: \text{Le}_S = \kappa / D_s

        """
        if self.salt_diffusivity == 0:
            return np.inf

        return self.thermal_diffusivity / self.salt_diffusivity
