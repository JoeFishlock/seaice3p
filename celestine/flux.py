import numpy as np
from celestine.grids import upwind, geometric


def calculate_conductive_heat_flux(temperature, D_g):
    r"""Calculate conductive heat flux as

    .. math:: -\frac{\partial\theta}{\partial z}

    :param temperature: temperature including ghost cells
    :type temperature: Numpy Array of size I+2
    :param D_g: difference matrix for ghost grid
    :type D_g: Numpy Array
    :return: conductive heat flux

    """
    return -np.matmul(D_g, temperature)


def calculate_advective_heat_flux(temperature, Wl):
    return upwind(temperature, Wl)


def calculate_frame_advection_heat_flux(enthalpy, V):
    return upwind(enthalpy, V)


def calculate_heat_flux(state_BCs, Wl, V, D_g):
    temperature = state_BCs.temperature
    enthalpy = state_BCs.enthalpy
    heat_flux = (
        calculate_conductive_heat_flux(temperature, D_g)
        + calculate_advective_heat_flux(temperature, Wl)
        + calculate_frame_advection_heat_flux(enthalpy, V)
    )
    return heat_flux


def calculate_diffusive_salt_flux(liquid_salinity, liquid_fraction, D_g, cfg):
    """Take liquid salinity and liquid fraction on ghost grid and interpolate liquid
    fraction geometrically"""
    lewis_salt = cfg.physical_params.lewis_salt
    return (
        -(1 / lewis_salt) * geometric(liquid_fraction) * np.matmul(D_g, liquid_salinity)
    )


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


def take_forward_euler_step(quantity, flux, timestep, D_e):
    r"""Advance the given quantity one forward Euler step using the given flux

    The quantity is given on cell centers and the flux on cell edges.

    Discretise the conservation equation

    .. math:: \frac{\partial Q}{\partial t} = -\frac{\partial F}{\partial z}

    as

    .. math:: Q^{n+1} = Q^n - \Delta t (\frac{\partial F}{\partial z})

    """
    return quantity - timestep * np.matmul(D_e, flux)
