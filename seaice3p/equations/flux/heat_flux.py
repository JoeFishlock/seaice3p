import numpy as np
from numpy.typing import NDArray

from ...grids import upwind, geometric
from ...params import Config, NoBrineConvection


def pure_liquid_switch(liquid_fraction: NDArray | float) -> NDArray | float:
    """Take the liquid fraction and return a smoothed switch that is equal to 1 in a
    pure liquid region and goes to zero rapidly outside of this"""
    SCALE = 1e-2
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


def calculate_advective_heat_flux(temperature, liquid_fraction, Wl, cfg):
    if isinstance(cfg.brine_convection_params, NoBrineConvection):
        return upwind(temperature, Wl)

    if cfg.brine_convection_params.advective_heat_flux_in_ocean:
        return upwind(temperature, Wl)

    # smoothly set advective heat transport in ocean to zero when using RJW14 brine convection
    # if no advective heat flux in the ocean
    # as the ocean should be turbulent and not drawing this additional heat flux
    return upwind(temperature, Wl) * (
        1 - pure_liquid_switch(geometric(liquid_fraction))
    )


def calculate_frame_advection_heat_flux(enthalpy, V):
    return upwind(enthalpy, V)


def calculate_heat_flux(state_BCs, Wl, V, D_g, cfg):
    temperature = state_BCs.temperature
    liquid_fraction = state_BCs.liquid_fraction
    enthalpy = state_BCs.enthalpy
    heat_flux = (
        calculate_conductive_heat_flux(state_BCs, D_g, cfg)
        + calculate_advective_heat_flux(temperature, liquid_fraction, Wl, cfg)
        + calculate_frame_advection_heat_flux(enthalpy, V)
    )
    return heat_flux
