#######################################################################
#                 Calculation for single bubble size                  #
#######################################################################

import numpy as np
from ...params import Config
from ...grids import geometric
from .bubble_parameters import calculate_bubble_size_fraction


def calculate_lag_function(bubble_size_fraction):
    r"""Calculate lag function from bubble size fraction on edge grid as

    .. math:: G(\lambda) = 1 - \lambda / 2

    for 0<lambda<1. Edge cases are given by G(0)=1 and G(1) = 0.5 for values outside
    this range.
    """
    lag = np.full_like(bubble_size_fraction, np.NaN)
    intermediate = (bubble_size_fraction < 1) & (bubble_size_fraction >= 0)
    large = bubble_size_fraction >= 1
    lag[bubble_size_fraction < 0] = 1
    lag[intermediate] = 1 - 0.5 * bubble_size_fraction[intermediate]
    lag[large] = 0.5
    return lag


def calculate_wall_drag_function(bubble_size_fraction, cfg: Config):
    r"""Calculate wall drag function from bubble size fraction on edge grid as

    .. math:: \frac{1}{K(\lambda)} = (1 - \lambda)^r

    in the power law case or in the Haberman case from the paper

    .. math:: \frac{1}{K(\lambda)} = \frac{1 -1.5\lambda + 1.5\lambda^5 - \lambda^6}{1+1.5\lambda^5}

    for 0<lambda<1. Edge cases are given by K(0)=1 and K(1) = 0 for values outside
    this range.
    """
    drag = np.full_like(bubble_size_fraction, np.NaN)
    intermediate = (bubble_size_fraction < 1) & (bubble_size_fraction >= 0)
    large = bubble_size_fraction >= 1
    drag[bubble_size_fraction < 0] = 1
    drag[intermediate] = (
        1
        - 1.5 * bubble_size_fraction[intermediate]
        + 1.5 * bubble_size_fraction[intermediate] ** 5
        - bubble_size_fraction[intermediate] ** 6
    ) / (1 + 1.5 * bubble_size_fraction[intermediate] ** 5)
    drag[large] = 0
    return drag


def calculate_mono_wall_drag_factor(liquid_fraction, cfg: Config):
    r"""Take liquid fraction on the ghost grid and calculate the wall drag factor
    for a mono bubble size distribution as

    .. math:: I_1 = \frac{\lambda^2}{K(\lambda)}

    returns wall drag factor on the edge grid
    """
    bubble_radius_scaled = cfg.bubble_params.bubble_radius_scaled
    bubble_size_fraction = calculate_bubble_size_fraction(
        bubble_radius_scaled, geometric(liquid_fraction), cfg
    )
    drag_function = calculate_wall_drag_function(bubble_size_fraction, cfg)
    drag_factor = drag_function * bubble_size_fraction**2
    return drag_factor


def calculate_mono_lag_factor(liquid_fraction, cfg: Config):
    r"""Take liquid fraction on the ghost grid and calculate the lag factor
    for a mono bubble size distribution as

    .. math:: I_2 = G(\lambda)

    returns lag factor on the edge grid
    """
    bubble_radius_scaled = cfg.bubble_params.bubble_radius_scaled
    bubble_size_fraction = calculate_bubble_size_fraction(
        bubble_radius_scaled, geometric(liquid_fraction), cfg
    )
    return calculate_lag_function(bubble_size_fraction)
