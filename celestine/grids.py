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


def average(ghosts):
    """Returns arithmetic mean pairwise of first dimension of an array

    This should get values on the ghost grid and returns the arithmetic average
    onto the edge grid
    """
    upper_ghosts = ghosts[1:]
    lower_ghosts = ghosts[:-1]
    return 0.5 * (upper_ghosts + lower_ghosts)


def add_ghost_cells(centers, bottom, top):
    """Add specified bottom and top value to center grid

    :param centers: numpy array on centered grid (size I).
    :type centers: Numpy array
    :param bottom: bottom value placed at index 0.
    :type bottom: float
    :param top: top value placed at index -1.
    :type top: float
    :return: numpy array on ghost grid (size I+2).
    """
    return np.concatenate((np.array([bottom]), centers, np.array([top])))


def centers_to_edges(centers, interpolator=geometric):
    """Takes quantity on center grid to cell edges.

    Top and bottom values are just top and bottom cell values. Middle values are
    interpolated. The default method for this is geometric.

    :param centers: numpy array on cell centers (size I).
    :type centers: Numpy array
    :param interpolator: function to interpolate cells to edges.
    :type interpolator: function with signature (ghosts) -> edges.
    :return: numpy array on cell edges (size I+1).
    """
    edges = np.full((centers.size + 1,), np.NaN)
    edges[0] = centers[0]
    edges[-1] = centers[-1]
    edges[1:-1] = interpolator(centers)
    return edges
