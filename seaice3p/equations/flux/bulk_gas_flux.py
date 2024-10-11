import numpy as np

from seaice3p.equations.flux.heat_flux import pure_liquid_switch
from ...grids import upwind, geometric
from ...params import Config


def calculate_diffusive_gas_flux(dissolved_gas, liquid_fraction, D_g, cfg: Config):
    chi = cfg.physical_params.expansion_coefficient
    lewis_gas = cfg.physical_params.lewis_gas
    edge_liquid_fraction = geometric(liquid_fraction)
    # Enhanced eddgy gas diffusivity in pure liquid region
    gas_diffusivity = (
        chi
        * edge_liquid_fraction
        * (
            (1 / lewis_gas)
            + cfg.physical_params.eddy_diffusivity_ratio
            * pure_liquid_switch(edge_liquid_fraction)
        )
    )
    return -gas_diffusivity * np.matmul(D_g, dissolved_gas)


def calculate_diffusive_gas_bubble_flux(
    gas_fraction, liquid_fraction, D_g, cfg: Config
):
    if not cfg.physical_params.gas_bubble_eddy_diffusion:
        return np.zeros_like(geometric(liquid_fraction))

    edge_liquid_fraction = geometric(liquid_fraction)
    # Enhanced eddgy gas diffusivity in pure liquid region
    gas_bubble_diffusivity = (
        cfg.physical_params.eddy_diffusivity_ratio
        * pure_liquid_switch(edge_liquid_fraction)
    )
    diffusive_flux = -gas_bubble_diffusivity * np.matmul(D_g, gas_fraction)
    diffusive_flux[-1] = 0
    return diffusive_flux


def calculate_bubble_gas_flux(gas_fraction, Vg):
    return upwind(gas_fraction, Vg)


def calculate_advective_dissolved_gas_flux(dissolved_gas, Wl, cfg):
    chi = cfg.physical_params.expansion_coefficient
    return chi * upwind(dissolved_gas, Wl)


def calculate_frame_advection_gas_flux(gas, V):
    return upwind(gas, V)


def calculate_gas_flux(state_BCs, Wl, V, Vg, D_g, cfg):
    dissolved_gas = state_BCs.dissolved_gas
    liquid_fraction = state_BCs.liquid_fraction
    gas_fraction = state_BCs.gas_fraction
    gas = state_BCs.gas
    gas_flux = (
        calculate_diffusive_gas_flux(dissolved_gas, liquid_fraction, D_g, cfg)
        + calculate_bubble_gas_flux(gas_fraction, Vg)
        + calculate_advective_dissolved_gas_flux(dissolved_gas, Wl, cfg)
        + calculate_frame_advection_gas_flux(gas, V)
        + calculate_diffusive_gas_bubble_flux(gas_fraction, liquid_fraction, D_g, cfg)
    )
    return gas_flux
