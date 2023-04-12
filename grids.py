import numpy as np


def get_number_of_timesteps(total_time, timestep):
    return int(total_time / timestep) + 1


def initialise_grids(number_of_cells):
    step = 1 / number_of_cells
    centers = np.array([-1 + (2 * i + 1) * step / 2 for i in range(number_of_cells)])
    edges = np.array([-1 + i * step for i in range(number_of_cells + 1)])
    ghosts = np.concatenate((np.array([-1 - step / 2]), centers, np.array([step / 2])))

    return step, centers, edges, ghosts


def get_difference_matrix(size, step):
    D = np.zeros((size, size + 1))
    for i in range(size):
        D[i, i] = -1
        D[i, i + 1] = 1
    return D / step


def upwind(ghosts, velocity):
    upper_ghosts = ghosts[1:]
    lower_ghosts = ghosts[:-1]
    upwards = np.maximum(velocity, 0)
    downwards = np.minimum(velocity, 0)
    edges = upwards * lower_ghosts + downwards * upper_ghosts
    return edges


def geometric(ghosts):
    """Returns geometric mean of the first dimension of an array"""
    upper_ghosts = ghosts[1:]
    lower_ghosts = ghosts[:-1]
    return np.sqrt(upper_ghosts * lower_ghosts)


def average(ghosts, params):
    A = np.zeros((params.I, params.I + 2))
    for i in range(params.I):
        A[i, i] = 0.5
        A[i, i + 2] = 0.5
    return np.matmul(A, ghosts)
