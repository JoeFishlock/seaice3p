import numpy as np

from seaice3p.equations.flux.heat_flux import pure_liquid_switch
from ...grids import upwind, geometric
from ...params import Config


def calculate_diffusive_salt_flux(liquid_salinity, liquid_fraction, D_g, cfg: Config):
    """Take liquid salinity and liquid fraction on ghost grid and interpolate liquid
    fraction geometrically"""
    lewis_salt = cfg.physical_params.lewis_salt
    edge_liquid_fraction = geometric(liquid_fraction)
    # In pure liquid phase enhanced eddy diffusivity of dissolved salt
    salt_diffusivity = edge_liquid_fraction * (
        (1 / lewis_salt)
        + cfg.physical_params.eddy_diffusivity_ratio
        * pure_liquid_switch(edge_liquid_fraction)
    )
    return -salt_diffusivity * np.matmul(D_g, liquid_salinity)


def calculate_advective_salt_flux(liquid_salinity, Wl, cfg):
    C = cfg.physical_params.concentration_ratio
    return upwind(liquid_salinity + C, Wl)


def calculate_frame_advection_salt_flux(salt, V):
    return upwind(salt, V)


def calculate_salt_flux(state_BCs, Wl, V, D_g, cfg):
    liquid_salinity = state_BCs.liquid_salinity
    liquid_fraction = state_BCs.liquid_fraction
    salt = state_BCs.salt
    salt_flux = (
        calculate_diffusive_salt_flux(liquid_salinity, liquid_fraction, D_g, cfg)
        + calculate_advective_salt_flux(liquid_salinity, Wl, cfg)
        + calculate_frame_advection_salt_flux(salt, V)
    )
    return salt_flux
