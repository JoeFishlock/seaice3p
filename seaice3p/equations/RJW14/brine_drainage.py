"""Module to calculate the Rees Jones and Worster 2014
parameterisation for brine convection velocity and the strenght of the sink term.

Exports the functions:

calculate_brine_convection_liquid_velocity
To be used in velocities module when using brine convection parameterisation.

calculate_brine_channel_sink
To be used to add sink terms to conservation equations when using brine convection
parameterisation.
"""


import numpy as np
from scipy.stats import hmean
from ...params import Config
from ...grids import calculate_ice_ocean_boundary_depth


def calculate_permeability(liquid_fraction, cfg: Config):
    r"""Calculate the absolute permeability as a function of liquid fraction

    .. math:: \Pi(\phi_l) = \phi_l^3

    Alternatively if the porosity threshold flag is true

    .. math:: \Pi(\phi_l) = \phi_l^2 (\phi_l - \phi_c)

    :param liquid_fraction: liquid fraction
    :type liquid_fraction: Numpy Array
    :param cfg: Configuration object for the simulation.
    :type cfg: seaice3p.params.Config
    :return: permeability on the same grid as liquid fraction
    """
    if cfg.bubble_params.porosity_threshold:
        cutoff = cfg.bubble_params.porosity_threshold_value
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
    :type cfg: seaice3p.params.Config
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
    :type cfg: seaice3p.params.Config
    :return: Array of shape (I,) of Rayleigh number at cell centers
    """
    Rayleigh_salt = cfg.brine_convection_params.Rayleigh_salt
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
    r"""Calculate the height of the convecting region as the top edge of the highest
    cell in the domain for which the quantity

    .. math:: \text{Ra}(z) - \text{Ra}_c

    is greater than or equal to zero.

    NOTE: if no convecting region exists return np.NaN

    :param Rayleigh_number: local rayleigh number on center grid
    :type Rayleigh_number: Numpy Array of shape (I,)
    :param edge_grid: The vertical coordinate positions of the edge grid.
    :type edge_grid: Numpy Array (size I+1)
    :param cfg: Configuration object for the simulation.
    :type cfg: seaice3p.params.Config
    :return: Edge grid value at convecting boundary.
    """
    Rayleigh_critical = cfg.brine_convection_params.Rayleigh_critical
    if np.all(Rayleigh_number - Rayleigh_critical < 0):
        return np.NaN
    indices = np.where(Rayleigh_number >= Rayleigh_critical)
    return edge_grid[indices[0][-1] + 1]


def get_effective_Rayleigh_number(Rayleigh_number, cfg: Config):
    r"""Calculate the effective Rayleigh Number as the maximum of

    .. math:: \text{Ra}(z) - \text{Ra}_c

    in the convecting region.

    NOTE: if no convecting region exists returns 0.

    :param Rayleigh_number: local rayleigh number on center grid
    :type Rayleigh_number: Numpy Array of shape (I,)
    :param cfg: Configuration object for the simulation.
    :type cfg: seaice3p.params.Config
    :return: Effective Rayleigh number.
    """
    Rayleigh_critical = cfg.brine_convection_params.Rayleigh_critical
    return np.max(
        np.where(
            Rayleigh_number >= Rayleigh_critical, Rayleigh_number - Rayleigh_critical, 0
        )
    )


def calculate_brine_channel_strength(
    Rayleigh_number, ice_depth, convecting_region_height, cfg: Config
):
    r"""Calculate the brine channel strength in the convecting region as

    .. math:: \mathcal{A} = \frac{\alpha \text{Ra}_e}{(h+z_c)^2}

    the effective Rayleigh number multiplied by a tuning parameter (Rees Jones and
    Worster 2014) over the convecting region thickness squared.

    :param Rayleigh_number: local Rayleigh number on center grid
    :type Rayleigh_number: Numpy Array of shape (I,)
    :param ice_depth: depth of ice (positive)
    :type ice_depth: float
    :param convecting_region_height: position of the convecting region boundary (negative)
    :type convecting_region_height: float
    :param cfg: Configuration object for the simulation.
    :type cfg: seaice3p.params.Config
    :return: Brine channel strength parameter
    """
    convection_strength = cfg.brine_convection_params.convection_strength
    if ice_depth == 0:
        return 0

    if np.isnan(convecting_region_height):
        return 0

    convecting_layer_thickness = ice_depth + convecting_region_height
    effective_Rayleigh = get_effective_Rayleigh_number(Rayleigh_number, cfg)
    return convection_strength * effective_Rayleigh / convecting_layer_thickness**2


def calculate_brine_convection_liquid_velocity(
    liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg: Config
):
    r"""Calculate the vertical liquid Darcy velocity from Rees Jones and Worster 2014

    .. math:: W_l = \mathcal{A} (z_c - z)

    in the convecting region. The velocity is stagnant above the convecting region.
    The velocity is constant in the liquid region and continuous at the interface.

    NOTE: If no ice is present or if no convecting region exists returns zero velocity

    :param liquid_fraction: liquid fraction on center grid
    :type liquid_fraction: Numpy Array of shape (I,)
    :param liquid_salinity: liquid salinity on center grid
    :type liquid_salinity: Numpy Array of shape (I,)
    :param center_grid: vertical coordinate of center grid
    :type center_grid: Numpy Array of shape (I,)
    :param edge_grid: Vertical coordinates of cell edges
    :type edge_grid: Numpy Array of shape (I+1,)
    :param cfg: Configuration object for the simulation.
    :type cfg: seaice3p.params.Config
    :return: Liquid darcy velocity on the edge grid.
    """
    ice_depth = calculate_ice_ocean_boundary_depth(liquid_fraction, edge_grid)
    Rayleigh_number = calculate_Rayleigh(
        center_grid, edge_grid, liquid_salinity, liquid_fraction, cfg
    )
    convecting_region_height = get_convecting_region_height(
        Rayleigh_number, edge_grid, cfg
    )
    brine_channel_strength = calculate_brine_channel_strength(
        Rayleigh_number, ice_depth, convecting_region_height, cfg
    )

    Wl = np.zeros_like(edge_grid)

    # No ice present
    if ice_depth == 0:
        return Wl

    # ice present but no convection occuring
    if np.isnan(convecting_region_height):
        return Wl

    # Make liquid vertical velocity continuous at bottom of the ice
    ocean_velocity = brine_channel_strength * (ice_depth + convecting_region_height)

    is_convecting_ice = (edge_grid >= -ice_depth) & (
        edge_grid <= convecting_region_height
    )
    is_liquid = edge_grid < -ice_depth

    Wl[is_convecting_ice] = brine_channel_strength * (
        convecting_region_height - edge_grid[is_convecting_ice]
    )
    Wl[is_liquid] = ocean_velocity

    return Wl


def calculate_brine_channel_sink(
    liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg: Config
):
    r"""Calculate the sink term due to brine channels.

    .. math:: \text{sink} = \mathcal{A}

    in the convecting region. Zero elsewhere.

    NOTE: If no ice is present or if no convecting region exists returns zero

    :param liquid_fraction: liquid fraction on center grid
    :type liquid_fraction: Numpy Array of shape (I,)
    :param liquid_salinity: liquid salinity on center grid
    :type liquid_salinity: Numpy Array of shape (I,)
    :param center_grid: vertical coordinate of center grid
    :type center_grid: Numpy Array of shape (I,)
    :param edge_grid: Vertical coordinates of cell edges
    :type edge_grid: Numpy Array of shape (I+1,)
    :param cfg: Configuration object for the simulation.
    :type cfg: seaice3p.params.Config
    :return: Strength of the sink term due to brine channels on the center grid.
    """
    ice_depth = calculate_ice_ocean_boundary_depth(liquid_fraction, edge_grid)
    Rayleigh_number = calculate_Rayleigh(
        center_grid, edge_grid, liquid_salinity, liquid_fraction, cfg
    )
    convecting_region_height = get_convecting_region_height(
        Rayleigh_number, edge_grid, cfg
    )
    brine_channel_strength = calculate_brine_channel_strength(
        Rayleigh_number, ice_depth, convecting_region_height, cfg
    )

    sink = np.zeros_like(center_grid)

    # No ice present
    if ice_depth == 0:
        return sink

    # ice present but no convection occuring
    if np.isnan(convecting_region_height):
        return sink

    # Make liquid vertical velocity continuous at bottom of the ice

    is_convecting_ice = (center_grid >= -ice_depth) & (
        center_grid <= convecting_region_height
    )
    is_liquid = center_grid < -ice_depth

    sink[is_convecting_ice] = brine_channel_strength
    sink[is_liquid] = 0

    return sink
