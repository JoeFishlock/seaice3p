import numpy as np
from grids import geometric
from phase_boundaries import calculate_eutectic

"""To prevent flow into a partially completely frozen region we must cut off
permeability if a cell becomes eutectic. I.e if H<H_E set perm=0 smoothly"""


def calculate_frame_velocity(params):
    return np.full((params.I + 1,), params.frame_velocity)


def calculate_absolute_permeability(liquid_fraction):
    return liquid_fraction**3


# def calculate_absolute_permeability(liquid_fraction, enthalpy, salt, gas, params):
#     boundary = calculate_eutectic(salt, gas, params)
#     return np.where(enthalpy <= boundary, 0, liquid_fraction**3)


def calculate_liquid_darcy_velocity(
    liquid_fraction, enthalpy, salt, gas, pressure, D_g, params
):
    # absolute_permeability = geometric(
    #     calculate_absolute_permeability(liquid_fraction, enthalpy, salt, gas, params)
    # )
    absolute_permeability = calculate_absolute_permeability(
        geometric(np.maximum(0, liquid_fraction - 0.146))
    )
    Wl = -absolute_permeability * np.matmul(D_g, pressure)
    return Wl


def calculate_bubble_radius(liquid_fraction, params):
    exponent = params.pore_throat_scaling
    reg = params.regularisation
    effective_tube_radius = geometric(liquid_fraction) ** exponent + reg
    # effective_tube_radius = (
    #     geometric(np.maximum(liquid_fraction - 0.2, 0)) ** exponent + reg
    # )
    return params.bubble_radius_scaled / effective_tube_radius


def calculate_lag(bubble_radius):
    lag = np.where(bubble_radius < 0, 1, 1 - 0.5 * bubble_radius)
    lag = np.where(bubble_radius > 1, 0.5, lag)
    return lag


def calculate_drag(bubble_radius, params):
    exponent = params.drag_exponent
    """release of gas during warming is sensitive to this exponent at least at lower buoyancy"""
    drag = np.where(bubble_radius < 0, 1, (1 - bubble_radius) ** exponent)
    drag = np.where(bubble_radius > 1, 0, drag)
    return drag


def calculate_gas_interstitial_velocity(
    liquid_fraction, enthalpy, salt, gas, pressure, D_g, params
):
    """For this to be on edge grid enter liquid fraction on edge grid"""
    B = params.B
    reg = params.regularisation
    Wl = calculate_liquid_darcy_velocity(
        liquid_fraction, enthalpy, salt, gas, pressure, D_g, params
    )
    bubble_radius = calculate_bubble_radius(liquid_fraction, params)
    drag = calculate_drag(bubble_radius, params)
    lag = calculate_lag(bubble_radius)

    # return B * drag + 2 * lag * Wl / (geometric(liquid_fraction) + reg)
    return B * drag


def calculate_velocities(liquid_fraction, enthalpy, salt, gas, pressure, D_g, params):
    Vg = calculate_gas_interstitial_velocity(
        liquid_fraction, enthalpy, salt, gas, pressure, D_g, params
    )
    Wl = calculate_liquid_darcy_velocity(
        liquid_fraction, enthalpy, salt, gas, pressure, D_g, params
    )
    V = calculate_frame_velocity(params)
    return Vg, Wl, V
