from dataclasses import dataclass
import numpy as np
from serde import serde, coerce
from scipy.optimize import fsolve


@serde(type_check=coerce)
@dataclass(frozen=True)
class LinearLiquidus:
    eutectic_temperature: float = -21.1  # deg Celsius
    eutectic_salinity: float = 270  # g/kg


@serde(type_check=coerce)
@dataclass(frozen=True)
class CubicLiquidus:
    """Cubic fit to liquidus to give liquidus salinity in terms of temperature

    S = a0 + a1 T + a2 T^2 + a3 T^3

    defaults are taken from Notz PhD thesis for fit to Assur seawater data
    """

    eutectic_temperature: float = -21.1  # deg Celsius
    a0: float = -1.2
    a1: float = -21.8
    a2: float = -0.919
    a3: float = -0.0178

    def get_liquidus_salinity(self, temperature):
        return (
            self.a0
            + self.a1 * temperature
            + self.a2 * temperature**2
            + self.a3 * temperature**3
        )

    def get_liquidus_temperature(self, salinity):
        temperature = fsolve(
            lambda x: salinity - self.get_liquidus_salinity(x),
            np.full_like(salinity, -2),
        )
        if temperature.size == 1:
            return temperature[0]
        else:
            return temperature


@serde(type_check=coerce)
@dataclass(frozen=True)
class DimensionalWaterParams:
    liquid_density: float = 1028  # kg/m3
    ice_density: float = 916  # kg/m3
    ocean_salinity: float = 34  # g/kg
    liquidus: LinearLiquidus | CubicLiquidus = LinearLiquidus()
    latent_heat: float = 334e3  # latent heat of fusion for ice in J/kg
    liquid_specific_heat_capacity: float = 4184  # J/kg degC
    solid_specific_heat_capacity: float = 2009  # J/kg degC
    liquid_thermal_conductivity: float = 0.54  # water thermal conductivity in W/m deg C
    solid_thermal_conductivity: float = 2.22  # ice thermal conductivity in W/m deg C
    snow_thermal_conductivity: float = 0.31  # snow thermal conductivity in W/m deg C
    snow_density: float = 150  # snow density kg/m3

    eddy_diffusivity: float = 0

    salt_diffusivity: float = 0  # molecular diffusivity of salt in water in m2/s
    # used to calculate Rayleigh number for convection and density contraction in liquid equation of state
    haline_contraction_coefficient: float = 7.5e-4  # 1/ppt

    # calculated from moreau et al 2014 value of kinematic viscosity for sewater 2.7e-6
    # dynamic liquid_viscosity = 2.7e-6 * liquid_density
    liquid_viscosity: float = 2.78e-3  # dynamic liquid viscosity in Pa.s

    @property
    def eutectic_salinity(self):
        if isinstance(self.liquidus, LinearLiquidus):
            return self.liquidus.eutectic_salinity
        if isinstance(self.liquidus, CubicLiquidus):
            return self.liquidus.get_liquidus_salinity(
                self.liquidus.eutectic_temperature
            )

        raise NotImplementedError

    @property
    def eutectic_temperature(self):
        if isinstance(self.liquidus, LinearLiquidus) or isinstance(
            self.liquidus, CubicLiquidus
        ):
            return self.liquidus.eutectic_temperature

        raise NotImplementedError

    @property
    def salinity_difference(self):
        r"""calculate difference between eutectic salinity and typical ocean salinity

        .. math:: \Delta S = S_E - S_i

        """
        return self.eutectic_salinity - self.ocean_salinity

    @property
    def ocean_freezing_temperature(self):
        """calculate salinity dependent freezing temperature using linear liquidus with
        ocean salinity

        .. math:: T_i = T_L(S_i) = T_E S_i / S_E

        or using a cubic fit for the liquidus curve

        """
        if isinstance(self.liquidus, LinearLiquidus):
            return (
                self.eutectic_temperature * self.ocean_salinity / self.eutectic_salinity
            )
        if isinstance(self.liquidus, CubicLiquidus):
            return self.liquidus.get_liquidus_temperature(self.ocean_salinity)

        raise NotImplementedError

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
