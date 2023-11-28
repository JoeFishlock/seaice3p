"""Module to calculate the Rees Jones and Worster 2014
parameterisation for brine convection velocity and the corresponding
fluxes of heat, salt, dissolved and free phase gas"""

import numpy as np
from scipy.stats import hmean
from celestine.params import Config


def calculate_permeability(liquid_fraction, cfg: Config):
    r"""Calculate the absolute permeability as a function of liquid fraction

    .. math:: \Pi(\phi_l) = \phi_l^3

    Alternatively if the porosity threshold flag is true

    .. math:: \Pi(\phi_l) = \phi_l^2 (\phi_l - \phi_c)

    :param liquid_fraction: liquid fraction
    :type liquid_fraction: Numpy Array
    :param cfg: Configuration object for the simulation.
    :type cfg: celestine.params.Config
    :return: permeability on the same grid as liquid fraction
    """
    if cfg.darcy_law_params.porosity_threshold:
        cutoff = cfg.darcy_law_params.porosity_threshold_value
        step_function = np.heaviside(liquid_fraction - cutoff, 0)
        return liquid_fraction**2 * (liquid_fraction - cutoff) * step_function
    return liquid_fraction**3


def calculate_integrated_mean_permeability(
    z, liquid_fraction, ice_depth, cell_centers, cfg: Config
):
    r"""Calculate the harmonic mean permeability from the base of the ice up to the
    cell containing the specified z value using the expression of ReesJones2014.

    .. math:: K(z) = (\frac{1}{h+z}\int_{-h}^{z} \frac{1}{\Pi(\phi_l(z'))}dz')^{-1}

    :param z: height to integrate permeability up to
    :type z: float
    :param liquid_fraction: liquid fraction on the center grid
    :type liquid_fraction: Numpy Array shape (I,)
    :param ice_depth: positive depth position of ice ocean interface
    :type ice_depth: float
    :param cell_centers: cell center positions
    :type cell_centers: Numpy Array of shape (I,)
    :param cfg: Configuration object for the simulation.
    :type cfg: celestine.params.Config
    :return: permeability averaged from base of the ice up to given z value
    """
    if z < -ice_depth:
        return 0
    step = cfg.numerical_params.step
    ice_mask = (cell_centers > -ice_depth) & (cell_centers <= z)
    permeabilities = (
        calculate_permeability(liquid_fraction[ice_mask], cfg)
        / liquid_fraction[ice_mask].size
    )
    harmonic_mean = hmean(permeabilities)
    return (ice_depth + z + step / 2) * harmonic_mean / step


def calculate_ice_ocean_boundary_depth(liquid_fraction, edge_grid):
    r"""Calculate the depth of the ice ocean boundary as the edge position of the
    first cell from the bottom to be not completely liquid. I.e the first time the
    liquid fraction goes below 1.

    If the ice has made it to the bottom of the domain raise an error.

    If the domain is completely liquid set h=0.

    NOTE: depth is a positive quantity and our grid coordinate increases from -1 at the
    bottom of the domain to 0 at the top.

    :param liquid_fraction: liquid fraction on center grid
    :type liquid_fraction: Numpy Array (size I)
    :param edge_grid: The vertical coordinate positions of the edge grid.
    :type edge_grid: Numpy Array (size I+1)
    :return: positive depth value of ice ocean interface
    """
    # locate index on center grid where liquid fraction first drops below 1
    index = np.argmax(liquid_fraction < 1)

    # if domain is completely liquid set h=0
    if np.all(liquid_fraction == 1):
        index = edge_grid.size - 1

    # raise error if bottom of domain freezes
    if index == 0:
        raise ValueError("Ice ocean interface has reached bottom of domain")

    # ice interface is at bottom edge of first frozen cell
    depth = (-1) * edge_grid[index]
    return depth


def calculate_Rayleigh(
    cell_centers, edge_grid, liquid_salinity, liquid_fraction, cfg: Config
):
    r"""Calculate the local Rayleigh number for brine convection as

    .. math:: \text{Ra}(z) = \text{Ra}_S K(z) (z+h) \Theta_l

    :param cell_centers: The vertical coordinates of cell centers.
    :type cell_centers: Numpy Array shape (I,)
    :param edge_grid: The vertical coordinate positions of the edge grid.
    :type edge_grid: Numpy Array (size I+1)
    :param liquid_salinity: liquid salinity on center grid
    :type liquid_salinity: Numpy Array shape (I,)
    :param liquid_fraction: liquid fraction on center grid
    :type liquid_fraction: Numpy Array (size I)
    :param cfg: Configuration object for the simulation.
    :type cfg: celestine.params.Config
    :return: Array of shape (I,) of Rayleigh number at cell centers
    """
    Rayleigh_salt = cfg.darcy_law_params.Rayleigh_salt
    ice_depth = calculate_ice_ocean_boundary_depth(liquid_fraction, edge_grid)
    averaged_permeabilities = np.array(
        [
            calculate_integrated_mean_permeability(
                z=z,
                liquid_fraction=liquid_fraction,
                ice_depth=ice_depth,
                cell_centers=cell_centers,
                cfg=cfg,
            )
            for z in cell_centers
        ]
    )
    return (
        Rayleigh_salt
        * (ice_depth + cell_centers)
        * averaged_permeabilities
        * liquid_salinity
    )


def get_convecting_region_height(Rayleigh_number, edge_grid, cfg: Config):
    Rayleigh_critical = cfg.darcy_law_params.Rayleigh_critical
    if np.all(Rayleigh_number - Rayleigh_critical < 0):
        return np.NaN
    indices = np.where(Rayleigh_number >= Rayleigh_critical)
    return edge_grid[indices[0][-1] + 1]


def get_effective_Rayleigh_number(Rayleigh_number, cfg: Config):
    Rayleigh_critical = cfg.darcy_law_params.Rayleigh_critical
    return np.max(
        np.where(
            Rayleigh_number >= Rayleigh_critical, Rayleigh_number - Rayleigh_critical, 0
        )
    )
