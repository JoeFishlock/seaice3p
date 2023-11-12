"""Module to calculate Darcy velocities.
"""

import numpy as np
from scipy.integrate import quad
from celestine.grids import geometric
from celestine.params import Config


def calculate_frame_velocity(cfg: Config):
    return np.full((cfg.numerical_params.I + 1,), cfg.physical_params.frame_velocity)


def calculate_liquid_darcy_velocity(liquid_fraction, cfg: Config):
    r"""Calculate liquid Darcy velocity as

    .. math:: W_l = \frac{\phi_l U_0}{2}

    This assumes that we are given the non dimensional maximum interstitial liquid
    velocity.

    :param liquid_fraction: liquid fraction on ghost grid
    :type liquid_fraction: Numpy Array (size I+2)
    :param cfg: simulation configuration object
    :type D_g: celestine.params.Config
    :return: liquid darcy velocity on edge grid
    """
    Wl = geometric(liquid_fraction) * cfg.darcy_law_params.liquid_velocity / 2
    return Wl


def calculate_bubble_size_fraction(bubble_radius_scaled, liquid_fraction, cfg: Config):
    r"""Takes bubble radius scaled and liquid fraction on edges and calculates the
    bubble size fraction as

    .. math:: \lambda = \Lambda / (\phi_l^q + \text{reg})

    Returns the bubble size fraction on the edge grid.
    """
    exponent = cfg.darcy_law_params.pore_throat_scaling
    reg = cfg.numerical_params.regularisation
    effective_tube_radius = liquid_fraction**exponent + reg
    return bubble_radius_scaled / effective_tube_radius


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

    for 0<lambda<1. Edge cases are given by K(0)=1 and K(1) = 0 for values outside
    this range.
    """
    exponent = cfg.darcy_law_params.drag_exponent
    drag = np.full_like(bubble_size_fraction, np.NaN)
    intermediate = (bubble_size_fraction < 1) & (bubble_size_fraction >= 0)
    large = bubble_size_fraction >= 1
    drag[bubble_size_fraction < 0] = 1
    drag[intermediate] = (1 - bubble_size_fraction[intermediate]) ** exponent
    drag[large] = 0
    return drag


def calculate_wall_drag_integrand(bubble_size_fraction: float, cfg: Config):
    r"""Scalar function to calculate wall drag integrand for polydispersive case.

    Bubble size fraction is given as a scalar input to calculate

    .. math:: \frac{\lambda^{5-p}}{K(\lambda)}

    """
    drag_exponent = cfg.darcy_law_params.drag_exponent
    power_law = cfg.darcy_law_params.bubble_distribution_power
    if bubble_size_fraction < 0:
        return 0
    elif (bubble_size_fraction >= 0) and (bubble_size_fraction < 1):
        return ((1 - bubble_size_fraction) ** drag_exponent) * (
            bubble_size_fraction ** (5 - power_law)
        )
    else:
        return 0


def calculate_lag_integrand(bubble_size_fraction: float, cfg: Config):
    r"""Scalar function to calculate lag integrand for polydispersive case.

    Bubble size fraction is given as a scalar input to calculate

    .. math:: \lambda^{3-p} G(\lambda)

    """
    drag_exponent = cfg.darcy_law_params.drag_exponent
    power_law = cfg.darcy_law_params.bubble_distribution_power
    if bubble_size_fraction < 0:
        return 0
    elif (bubble_size_fraction >= 0) and (bubble_size_fraction < 1):
        return (1 - 0.5 * bubble_size_fraction) * (
            bubble_size_fraction ** (3 - power_law)
        )
    else:
        return 0.5


def calculate_volume_integrand(bubble_size_fraction: float, cfg: Config):
    p = cfg.darcy_law_params.bubble_distribution_power
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
        cfg.darcy_law_params.minimum_bubble_radius_scaled,
        geometric(liquid_fraction),
        cfg,
    )
    maximum_size_fractions = calculate_bubble_size_fraction(
        cfg.darcy_law_params.maximum_bubble_radius_scaled,
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
        cfg.darcy_law_params.minimum_bubble_radius_scaled,
        geometric(liquid_fraction),
        cfg,
    )
    maximum_size_fractions = calculate_bubble_size_fraction(
        cfg.darcy_law_params.maximum_bubble_radius_scaled,
        geometric(liquid_fraction),
        cfg,
    )
    lag_factor = np.full_like(minimum_size_fractions, np.NaN)
    for i, (min, max) in enumerate(zip(minimum_size_fractions, maximum_size_fractions)):
        lag_factor[i] = calculate_lag_integral(min, max, cfg)
    return lag_factor


def calculate_mono_wall_drag_factor(liquid_fraction, cfg: Config):
    r"""Take liquid fraction on the ghost grid and calculate the wall drag factor
    for a mono bubble size distribution as

    .. math:: I_1 = \frac{\lambda^2}{K(\lambda)}

    returns wall drag factor on the edge grid
    """
    bubble_radius_scaled = cfg.darcy_law_params.bubble_radius_scaled
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
    bubble_radius_scaled = cfg.darcy_law_params.bubble_radius_scaled
    bubble_size_fraction = calculate_bubble_size_fraction(
        bubble_radius_scaled, geometric(liquid_fraction), cfg
    )
    return calculate_lag_function(bubble_size_fraction)


def calculate_gas_interstitial_velocity(
    liquid_fraction,
    liquid_interstitial_velocity,
    wall_drag_factor,
    lag_factor,
    cfg: Config,
):
    r"""Calculate Vg from liquid fraction on the ghost frid and liquid interstitial velocity

    .. math:: V_g = \mathcal{B} (\phi_l^{2q} I_1) + U_0 I_2

    Return Vg on edge grid
    """
    B = cfg.darcy_law_params.B
    exponent = cfg.darcy_law_params.pore_throat_scaling

    return (
        B * wall_drag_factor * geometric(liquid_fraction) ** (2 * exponent)
        + liquid_interstitial_velocity * lag_factor
    )


def calculate_velocities(state_BCs, cfg: Config):
    "Inputs on ghost grid, outputs on edge grid" ""
    liquid_fraction = state_BCs.liquid_fraction
    liquid_interstitial_velocity = cfg.darcy_law_params.liquid_velocity

    if cfg.darcy_law_params.bubble_size_distribution_type == "mono":
        wall_drag_factor = calculate_mono_wall_drag_factor(liquid_fraction, cfg)
        lag_factor = calculate_mono_lag_factor(liquid_fraction, cfg)
    elif cfg.darcy_law_params.bubble_size_distribution_type == "power_law":
        wall_drag_factor = calculate_power_law_wall_drag_factor(liquid_fraction, cfg)
        lag_factor = calculate_power_law_lag_factor(liquid_fraction, cfg)
    else:
        raise ValueError(
            f"Bubble size distribution of type {cfg.darcy_law_params.bubble_size_distribution_type} not recognised"
        )

    Vg = calculate_gas_interstitial_velocity(
        liquid_fraction, liquid_interstitial_velocity, wall_drag_factor, lag_factor, cfg
    )
    Wl = calculate_liquid_darcy_velocity(liquid_fraction, cfg)
    V = calculate_frame_velocity(cfg)
    return Vg, Wl, V
