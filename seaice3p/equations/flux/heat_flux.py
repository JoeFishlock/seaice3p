import numpy as np
from ...grids import upwind, geometric


def calculate_conductive_heat_flux(state_BCs, D_g, cfg):
    r"""Calculate conductive heat flux as

    .. math:: -\frac{\partial\theta}{\partial z}

    or alteratively if the phase_average_conductivity configuration parameter
    is set to True then we use the conductivity ratio as follows

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
    if not cfg.physical_params.phase_average_conductivity:
        return -np.matmul(D_g, temperature)

    conductivity_ratio = cfg.physical_params.conductivity_ratio
    edge_liquid_fraction = geometric(state_BCs.liquid_fraction)
    edge_solid_fraction = 1 - edge_liquid_fraction
    return -(
        edge_liquid_fraction + conductivity_ratio * edge_solid_fraction
    ) * np.matmul(D_g, temperature)


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
