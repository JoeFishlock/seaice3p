"""Module to calculate Darcy velocities.
"""

import numpy as np
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


def calculate_lag(bubble_radius):
    lag = np.where(bubble_radius < 0, 1, 1 - 0.5 * bubble_radius)
    lag = np.where(bubble_radius > 1, 0.5, lag)
    return lag


def calculate_drag(bubble_radius, cfg: Config):
    exponent = cfg.darcy_law_params.drag_exponent
    """release of gas during warming is sensitive to this exponent at least at lower buoyancy"""
    drag = np.where(bubble_radius < 0, 1, (1 - bubble_radius) ** exponent)
    drag = np.where(bubble_radius > 1, 0, drag)
    return drag


def calculate_gas_interstitial_velocity(liquid_fraction, cfg: Config):
    r"""Calculate Vg from liquid fraction and liquid Darcy velocity

    .. math:: V_g = \mathcal{B} (\phi_l^{2q} \frac{\lambda^2}{K(\lambda)}) + U_0 G(\lambda)

    Return Vg on edge grid"""
    B = cfg.darcy_law_params.B

    single_bubble_scaled = cfg.darcy_law_params.bubble_radius_scaled
    bubble_radius = calculate_bubble_size_fraction(
        single_bubble_scaled, geometric(liquid_fraction), cfg
    )
    drag = calculate_drag(bubble_radius, cfg)
    pore_throat_scaling = cfg.darcy_law_params.pore_throat_scaling

    # reg = cfg.numerical_params.regularisation
    # lag = calculate_lag(bubble_radius)
    # Wl = calculate_liquid_darcy_velocity(liquid_fraction, pressure, D_g)
    # return B * drag * (
    #     geometric(liquid_fraction) ** (2 * pore_throat_scaling)
    # ) * bubble_radius**2 + 2 * lag * Wl / (geometric(liquid_fraction) + reg)
    return (
        B
        * drag
        * (geometric(liquid_fraction) ** (2 * pore_throat_scaling))
        * bubble_radius**2
    )


def calculate_velocities(state_BCs, cfg: Config):
    "Inputs on ghost grid, outputs on edge grid" ""
    liquid_fraction = state_BCs.liquid_fraction
    Vg = calculate_gas_interstitial_velocity(liquid_fraction, cfg)
    Wl = calculate_liquid_darcy_velocity(liquid_fraction, cfg)
    V = calculate_frame_velocity(cfg)
    return Vg, Wl, V
