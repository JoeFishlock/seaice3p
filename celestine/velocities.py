import numpy as np
from celestine.grids import geometric
from celestine.phase_boundaries import calculate_eutectic
from celestine.params import Config

"""To prevent flow into a partially completely frozen region we must cut off
permeability if a cell becomes eutectic. I.e if H<H_E set perm=0 smoothly"""


def calculate_frame_velocity(cfg: Config):
    return np.full((cfg.numerical_params.I + 1,), cfg.physical_params.frame_velocity)


def calculate_absolute_permeability(liquid_fraction):
    return liquid_fraction**3


# def calculate_absolute_permeability(liquid_fraction, enthalpy, salt, gas, cfg: Config):
#     boundary = calculate_eutectic(salt, gas, cfg)
#     return np.where(enthalpy <= boundary, 0, liquid_fraction**3)


def calculate_liquid_darcy_velocity(
    liquid_fraction, enthalpy, salt, gas, pressure, D_g, cfg: Config
):
    # absolute_permeability = geometric(
    #     calculate_absolute_permeability(liquid_fraction, enthalpy, salt, gas, cfg)
    # )
    absolute_permeability = calculate_absolute_permeability(
        geometric(np.maximum(0, liquid_fraction - 0.146))
    )
    Wl = -absolute_permeability * np.matmul(D_g, pressure)
    return Wl


def calculate_bubble_radius(liquid_fraction, cfg: Config):
    exponent = cfg.darcy_law_params.pore_throat_scaling
    reg = cfg.numerical_params.regularisation
    effective_tube_radius = geometric(liquid_fraction) ** exponent + reg
    # effective_tube_radius = (
    #     geometric(np.maximum(liquid_fraction - 0.2, 0)) ** exponent + reg
    # )
    return cfg.darcy_law_params.bubble_radius_scaled / effective_tube_radius


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


def calculate_gas_interstitial_velocity(
    liquid_fraction, enthalpy, salt, gas, pressure, D_g, cfg: Config
):
    """For this to be on edge grid enter liquid fraction on edge grid"""
    B = cfg.darcy_law_params.B
    reg = cfg.numerical_params.regularisation
    Wl = calculate_liquid_darcy_velocity(
        liquid_fraction, enthalpy, salt, gas, pressure, D_g, cfg
    )
    bubble_radius = calculate_bubble_radius(liquid_fraction, cfg)
    drag = calculate_drag(bubble_radius, cfg)
    lag = calculate_lag(bubble_radius)

    # return B * drag + 2 * lag * Wl / (geometric(liquid_fraction) + reg)
    return B * drag


def calculate_velocities(
    liquid_fraction, enthalpy, salt, gas, pressure, D_g, cfg: Config
):
    Vg = calculate_gas_interstitial_velocity(
        liquid_fraction, enthalpy, salt, gas, pressure, D_g, cfg
    )
    Wl = calculate_liquid_darcy_velocity(
        liquid_fraction, enthalpy, salt, gas, pressure, D_g, cfg
    )
    V = calculate_frame_velocity(cfg)
    return Vg, Wl, V
