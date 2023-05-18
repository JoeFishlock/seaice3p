import numpy as np
from celestine.grids import upwind, geometric
from celestine.params import Config

"""To prevent flow into a partially completely frozen region we must cut off
permeability if a cell becomes eutectic. I.e if H<H_E set perm=0 smoothly"""


def calculate_frame_velocity(cfg: Config):
    return np.full((cfg.numerical_params.I + 1,), cfg.physical_params.frame_velocity)


def calculate_absolute_permeability(liquid_fraction):
    return liquid_fraction**3


def calculate_liquid_darcy_velocity(liquid_fraction, pressure, D_g):
    r"""Calculate liquid Darcy velocity as

    .. math:: W_l = -\Pi(\phi_l) \frac{\partial p}{\partial z}

    :param liquid_fraction: liquid fraction on ghost grid
    :type liquid_fraction: Numpy Array (size I+2)
    :param pressure: pressure on ghost grid
    :type pressure: Numpy Array (size I+2)
    :param D_g: difference matrix for ghost grid
    :type D_g: Numpy Array (size I+2)
    :return: liquid darcy velocity on edge grid
    """
    absolute_permeability = geometric(calculate_absolute_permeability(liquid_fraction))
    Wl = -absolute_permeability * np.matmul(D_g, pressure)
    return Wl


def solve_pressure_equation(state_BCs, new_state_BCs, timestep, D_e, D_g, cfg: Config):
    """Calculate pressure on ghost grid from current and new state on ghost grid

    Return new pressure on centers but easy to add boundary conditions"""
    I = cfg.numerical_params.I
    V = cfg.physical_params.frame_velocity

    permeability = geometric(
        calculate_absolute_permeability(new_state_BCs.liquid_fraction)
    )

    pressure_matrix = np.zeros((I + 2, I + 2))
    perm_matrix = np.zeros((I + 1, I + 1))
    pressure_forcing = np.zeros((I + 2,))

    new_gas_fraction = new_state_BCs.gas_fraction
    gas_fraction = state_BCs.gas_fraction
    pressure_forcing[1:-1] = (1 / timestep) * (
        new_gas_fraction[1:-1] - gas_fraction[1:-1]
    ) + np.matmul(D_e, upwind(new_gas_fraction, V))

    np.fill_diagonal(perm_matrix, permeability + 1e-4)
    pressure_matrix[1:-1, :] = np.matmul(D_e, np.matmul(-perm_matrix, D_g))
    pressure_matrix[0, 0] = 1
    pressure_matrix[-1, -1] = 1
    pressure_matrix[-1, -2] = -1

    new_pressure = np.linalg.solve(pressure_matrix, pressure_forcing)

    # return the new pressure on centers
    return new_pressure


def calculate_bubble_radius(liquid_fraction, cfg: Config):
    """Takes liquid fraction on edges and returns bubble radius parameter on edges"""
    exponent = cfg.darcy_law_params.pore_throat_scaling
    reg = cfg.numerical_params.regularisation
    effective_tube_radius = liquid_fraction**exponent + reg
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


def calculate_gas_interstitial_velocity(liquid_fraction, pressure, D_g, cfg: Config):
    """Calculate Vg from liquid fraction and pressure on ghost grid

    Return Vg on edge grid"""
    B = cfg.darcy_law_params.B

    bubble_radius = calculate_bubble_radius(geometric(liquid_fraction), cfg)
    drag = calculate_drag(bubble_radius, cfg)

    # reg = cfg.numerical_params.regularisation
    # lag = calculate_lag(bubble_radius)
    # Wl = calculate_liquid_darcy_velocity(liquid_fraction, pressure, D_g)
    # return B * drag + 2 * lag * Wl / (geometric(liquid_fraction) + reg)
    return B * drag


def calculate_velocities(state_BCs, D_g, cfg: Config):
    "Inputs on ghost grid, outputs on edge grid" ""
    liquid_fraction = state_BCs.liquid_fraction
    pressure = state_BCs.pressure
    Vg = calculate_gas_interstitial_velocity(liquid_fraction, pressure, D_g, cfg)
    Wl = calculate_liquid_darcy_velocity(liquid_fraction, pressure, D_g)
    V = calculate_frame_velocity(cfg)
    return Vg, Wl, V
