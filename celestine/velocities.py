import numpy as np
from celestine.grids import geometric, upwind, centers_to_edges, add_ghost_cells
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

    :param liquid_fraction: liquid fraction on center grid
    :type liquid_fraction: Numpy Array (size I)
    :param pressure: pressure on center grid
    :type pressure: Numpy Array (size I)
    :param D_g: difference matrix for ghost grid
    :type D_g: Numpy Array (size I+2)
    :return: liquid darcy velocity on edge grid
    """
    liquid_fraction_on_edges = centers_to_edges(liquid_fraction)
    absolute_permeability = calculate_absolute_permeability(liquid_fraction_on_edges)
    pressure_ghost = add_ghost_cells(pressure, 0, pressure[-1])  # BCs for Wl=0 at top
    Wl = -absolute_permeability * np.matmul(D_g, pressure)
    return Wl


def solve_pressure_equation(
    gas_fraction, new_gas_fraction, permeability, timestep, D_e, D_g, cfg: Config
):
    """Calculate pressure on center grid from gas fractions on center grid and
    permeability on edge grid

    Return new pressure on centers but easy to add boundary conditions"""
    I = cfg.numerical_params.I
    V = cfg.physical_params.frame_velocity

    pressure_matrix = np.zeros((I + 2, I + 2))
    perm_matrix = np.zeros((I + 1, I + 1))
    pressure_forcing = np.zeros((I + 2,))

    new_gas_fraction_ghosts = add_ghost_cells(new_gas_fraction, 0, 0)
    pressure_forcing[1:-1] = (1 / timestep) * (
        new_gas_fraction - gas_fraction
    ) + np.matmul(D_e, upwind(new_gas_fraction_ghosts, V))

    np.fill_diagonal(perm_matrix, permeability + 1e-4)
    pressure_matrix[1:-1, :] = np.matmul(D_e, np.matmul(-perm_matrix, D_g))
    pressure_matrix[0, 0] = 1
    pressure_matrix[-1, -1] = 1
    pressure_matrix[-1, -2] = -1

    new_pressure = np.linalg.solve(pressure_matrix, pressure_forcing)

    # return the new pressure on centers
    return new_pressure[1:-1]


def calculate_bubble_radius(liquid_fraction, cfg: Config):
    exponent = cfg.darcy_law_params.pore_throat_scaling
    reg = cfg.numerical_params.regularisation
    effective_tube_radius = geometric(liquid_fraction) ** exponent + reg
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
    """Calculate Vg from liquid fraction and pressure on center grid

    Return Vg on edge grid"""
    B = cfg.darcy_law_params.B
    reg = cfg.numerical_params.regularisation

    Wl = calculate_liquid_darcy_velocity(liquid_fraction, pressure, D_g)
    bubble_radius = calculate_bubble_radius(centers_to_edges(liquid_fraction), cfg)
    drag = calculate_drag(bubble_radius, cfg)
    lag = calculate_lag(bubble_radius)

    # return B * drag + 2 * lag * Wl / (geometric(liquid_fraction) + reg)
    return B * drag


def calculate_velocities(liquid_fraction, pressure, D_g, cfg: Config):
    "Inputs on center grid, outputs on edge grid" ""
    Vg = calculate_gas_interstitial_velocity(liquid_fraction, pressure, D_g, cfg)
    Wl = calculate_liquid_darcy_velocity(liquid_fraction, pressure, D_g)
    V = calculate_frame_velocity(cfg)
    return Vg, Wl, V
