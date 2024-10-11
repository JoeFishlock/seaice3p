import numpy as np
from numpy.typing import NDArray
from ...grids import upwind, geometric
from ...params import Config


def pure_liquid_switch(liquid_fraction: NDArray | float) -> NDArray | float:
    """Take the liquid fraction and return a smoothed switch that is equal to 1 in a
    pure liquid region and goes to zero rapidly outside of this"""
    SCALE = 5e-3
    return np.exp((liquid_fraction - 1) / SCALE)


def calculate_conductivity(
    cfg: Config, solid_fraction: NDArray | float
) -> NDArray | float:
    liquid_fraction = 1 - solid_fraction

    return (
        liquid_fraction
        + cfg.physical_params.conductivity_ratio * solid_fraction
        + cfg.physical_params.eddy_diffusivity_ratio
        * pure_liquid_switch(liquid_fraction)
    )


def calculate_conductive_heat_flux(state_BCs, D_g, cfg):
    r"""Calculate conductive heat flux as

    .. math:: -[(\phi_l + \lambda \phi_s) \frac{\partial \theta}{\partial z}]

    :param temperature: temperature including ghost cells
    :type temperature: Numpy Array of size I+2
    :param D_g: difference matrix for ghost grid
    :type D_g: Numpy Array
    :param cfg: Simulation configuration
    :type cfg: seaice3p.params.Config
    :return: conductive heat flux

    """
    temperature = state_BCs.temperature
    edge_liquid_fraction = geometric(state_BCs.liquid_fraction)
    edge_solid_fraction = 1 - edge_liquid_fraction
    conductivity = calculate_conductivity(cfg, edge_solid_fraction)
    return -conductivity * np.matmul(D_g, temperature)


def calculate_advective_heat_flux(temperature, Wl):
    return upwind(temperature, Wl)


def calculate_frame_advection_heat_flux(enthalpy, V):
    return upwind(enthalpy, V)


def calculate_heat_flux(state_BCs, Wl, V, D_g, cfg):
    temperature = state_BCs.temperature
    enthalpy = state_BCs.enthalpy
    heat_flux = (
        calculate_conductive_heat_flux(state_BCs, D_g, cfg)
        + calculate_advective_heat_flux(temperature, Wl)
        + calculate_frame_advection_heat_flux(enthalpy, V)
    )
    return heat_flux
