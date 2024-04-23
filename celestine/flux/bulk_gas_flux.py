import numpy as np
from ..grids import upwind, geometric


def calculate_diffusive_gas_flux(dissolved_gas, liquid_fraction, D_g, cfg):
    chi = cfg.physical_params.expansion_coefficient
    lewis_gas = cfg.physical_params.lewis_gas
    return (
        -(chi / lewis_gas) * geometric(liquid_fraction) * np.matmul(D_g, dissolved_gas)
    )


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
    )
    return gas_flux
