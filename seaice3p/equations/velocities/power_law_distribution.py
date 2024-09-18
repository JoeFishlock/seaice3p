#######################################################################
#             Calculation for power law size distribution             #
#######################################################################

import numpy as np
from scipy.integrate import quad
from ...params import Config
from ...grids import geometric
from .bubble_parameters import calculate_bubble_size_fraction


def calculate_wall_drag_integrand(bubble_size_fraction: float, cfg: Config):
    r"""Scalar function to calculate wall drag integrand for polydispersive case.

    Bubble size fraction is given as a scalar input to calculate

    .. math:: \frac{\lambda^{5-p}}{K(\lambda)}

    where the wall drag enhancement funciton K can be given by a power law fit
    or taken from the Haberman paper.
    """
    power_law = cfg.bubble_params.bubble_distribution_power
    if bubble_size_fraction < 0:
        return 0
    elif (bubble_size_fraction >= 0) and (bubble_size_fraction < 1):
        return (
            (
                1
                - 1.5 * bubble_size_fraction
                + 1.5 * bubble_size_fraction**5
                - bubble_size_fraction**6
            )
            / (1 + 1.5 * bubble_size_fraction**5)
        ) * (bubble_size_fraction ** (5 - power_law))
    else:
        return 0


def calculate_lag_integrand(bubble_size_fraction: float, cfg: Config):
    r"""Scalar function to calculate lag integrand for polydispersive case.

    Bubble size fraction is given as a scalar input to calculate

    .. math:: \lambda^{3-p} G(\lambda)

    """
    power_law = cfg.bubble_params.bubble_distribution_power
    if bubble_size_fraction < 0:
        return 0
    elif (bubble_size_fraction >= 0) and (bubble_size_fraction < 1):
        return (1 - 0.5 * bubble_size_fraction) * (
            bubble_size_fraction ** (3 - power_law)
        )
    else:
        return 0.5


def calculate_volume_integrand(bubble_size_fraction: float, cfg: Config):
    r"""Scalar function to calculate the integrand for volume under a power law
    bubble size distribution given as

    .. math:: \lambda^{3-p}

    in terms of the bubble size fraction.
    """
    p = cfg.bubble_params.bubble_distribution_power
    return bubble_size_fraction ** (3 - p)


def calculate_wall_drag_integral(
    bubble_size_fraction_min: float, bubble_size_fraction_max: float, cfg: Config
):
    numerator = quad(
        lambda x: calculate_wall_drag_integrand(x, cfg),
        bubble_size_fraction_min,
        bubble_size_fraction_max,
    )[0]
    denominator = quad(
        lambda x: calculate_volume_integrand(x, cfg),
        bubble_size_fraction_min,
        bubble_size_fraction_max,
    )[0]
    return numerator / denominator


def calculate_lag_integral(
    bubble_size_fraction_min: float, bubble_size_fraction_max: float, cfg: Config
):
    numerator = quad(
        lambda x: calculate_lag_integrand(x, cfg),
        bubble_size_fraction_min,
        bubble_size_fraction_max,
    )[0]
    denominator = quad(
        lambda x: calculate_volume_integrand(x, cfg),
        bubble_size_fraction_min,
        bubble_size_fraction_max,
    )[0]
    return numerator / denominator


def calculate_power_law_wall_drag_factor(liquid_fraction, cfg: Config):
    r"""Take liquid fraction on the ghost grid and calculate the wall drag factor
    for power law bubble size distribution.

    Return on edge grid
    """
    minimum_size_fractions = calculate_bubble_size_fraction(
        cfg.bubble_params.minimum_bubble_radius_scaled,
        geometric(liquid_fraction),
        cfg,
    )
    maximum_size_fractions = calculate_bubble_size_fraction(
        cfg.bubble_params.maximum_bubble_radius_scaled,
        geometric(liquid_fraction),
        cfg,
    )
    drag_factor = np.full_like(minimum_size_fractions, np.NaN)
    for i, (min, max) in enumerate(zip(minimum_size_fractions, maximum_size_fractions)):
        drag_factor[i] = calculate_wall_drag_integral(min, max, cfg)
    return drag_factor


def calculate_power_law_lag_factor(liquid_fraction, cfg: Config):
    r"""Take liquid fraction on the ghost grid and calculate the lag factor
    for power law bubble size distribution.

    Return on edge grid
    """
    minimum_size_fractions = calculate_bubble_size_fraction(
        cfg.bubble_params.minimum_bubble_radius_scaled,
        geometric(liquid_fraction),
        cfg,
    )
    maximum_size_fractions = calculate_bubble_size_fraction(
        cfg.bubble_params.maximum_bubble_radius_scaled,
        geometric(liquid_fraction),
        cfg,
    )
    lag_factor = np.full_like(minimum_size_fractions, np.NaN)
    for i, (min, max) in enumerate(zip(minimum_size_fractions, maximum_size_fractions)):
        lag_factor[i] = calculate_lag_integral(min, max, cfg)
    return lag_factor
