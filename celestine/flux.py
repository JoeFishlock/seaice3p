import numpy as np
from celestine.grids import upwind


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
