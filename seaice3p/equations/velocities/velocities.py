import numpy as np
from ...grids import geometric, Grids
from ...params import (
    Config,
    MonoBubbleParams,
    PowerLawBubbleParams,
    NoBrineConvection,
)
from ..RJW14 import calculate_brine_convection_liquid_velocity
from .mono_distribution import (
    calculate_mono_wall_drag_factor,
    calculate_mono_lag_factor,
)
from .power_law_distribution import (
    calculate_power_law_wall_drag_factor,
    calculate_power_law_lag_factor,
)


def calculate_frame_velocity(cfg: Config):
    return np.full((cfg.numerical_params.I + 1,), cfg.physical_params.frame_velocity)


def calculate_liquid_darcy_velocity(
    liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg: Config
):
    r"""Calculate liquid Darcy velocity either using brine convection parameterisation
    or as stagnant


    :param liquid_fraction: liquid fraction on ghost grid
    :type liquid_fraction: Numpy Array (size I+2)
    :param liquid_salinity: liquid salinity on ghost grid
    :type liquid_salinity: Numpy Array (size I+2)
    :param center_grid: vertical coordinates of cell centers
    :type center_grid: Numpy Array of shape (I,)
    :param edge_grid: Vertical coordinates of cell edges
    :type edge_grid: Numpy Array (size I+1)
    :param cfg: simulation configuration object
    :type cfg: seaice3p.params.Config
    :return: liquid darcy velocity on edge grid
    """
    if isinstance(cfg.brine_convection_params, NoBrineConvection):
        return np.zeros_like(geometric(liquid_fraction))

    Wl = calculate_brine_convection_liquid_velocity(
        liquid_fraction[1:-1], liquid_salinity[1:-1], center_grid, edge_grid, cfg
    )
    return Wl


def calculate_gas_interstitial_velocity(
    liquid_fraction,
    liquid_darcy_velocity,
    wall_drag_factor,
    lag_factor,
    cfg: Config,
):
    r"""Calculate Vg from liquid fraction on the ghost frid and liquid interstitial velocity

    .. math:: V_g = \mathcal{B} (\phi_l^{2q} I_1) + U_0 I_2

    Return Vg on edge grid
    """
    B = cfg.bubble_params.B
    exponent = cfg.bubble_params.pore_throat_scaling

    REGULARISATION = 1e-10
    liquid_interstitial_velocity = (
        liquid_darcy_velocity * 2 / (geometric(liquid_fraction) + REGULARISATION)
    )

    viscosity_factor = (
        2
        * (1 + cfg.physical_params.gas_viscosity_ratio)
        / (2 + 3 * cfg.physical_params.gas_viscosity_ratio)
    )
    Vg = (
        viscosity_factor
        * B
        * wall_drag_factor
        * geometric(liquid_fraction) ** (2 * exponent)
        + liquid_interstitial_velocity * lag_factor
    )

    # apply a porosity cutoff to the gas interstitial velocity if necking occurs below
    # critical porosity.
    if cfg.bubble_params.porosity_threshold:
        return Vg * np.heaviside(
            geometric(liquid_fraction) - cfg.bubble_params.porosity_threshold_value,
            0,
        )

    return Vg


def calculate_velocities(state_BCs, cfg: Config):
    """Inputs on ghost grid, outputs on edge grid

    needs the simulation config, liquid fraction, liquid salinity and grids
    """
    liquid_fraction = state_BCs.liquid_fraction
    liquid_salinity = state_BCs.liquid_salinity
    center_grid, edge_grid = (
        Grids(cfg.numerical_params.I).centers,
        Grids(cfg.numerical_params.I).edges,
    )

    match cfg.bubble_params:
        case MonoBubbleParams():
            wall_drag_factor = calculate_mono_wall_drag_factor(liquid_fraction, cfg)
            lag_factor = calculate_mono_lag_factor(liquid_fraction, cfg)
        case PowerLawBubbleParams():
            wall_drag_factor = calculate_power_law_wall_drag_factor(
                liquid_fraction, cfg
            )
            lag_factor = calculate_power_law_lag_factor(liquid_fraction, cfg)
        case _:
            raise NotImplementedError

    # check if we want to couple the bubble to fluid motion in the vertical
    if not isinstance(cfg.brine_convection_params, NoBrineConvection):
        if not cfg.brine_convection_params.couple_bubble_to_vertical_flow:
            lag_factor = np.zeros_like(wall_drag_factor)

    Wl = calculate_liquid_darcy_velocity(
        liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg
    )
    Vg = calculate_gas_interstitial_velocity(
        liquid_fraction, Wl, wall_drag_factor, lag_factor, cfg
    )
    V = calculate_frame_velocity(cfg)
    return Vg, Wl, V
