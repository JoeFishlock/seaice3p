"""Module providing functions to initialise the different grids and interpolate
quantities between them.
"""

from dataclasses import dataclass
from functools import cached_property
import numpy as np
from numpy.typing import NDArray


def get_difference_matrix(size, step):
    D = np.zeros((size, size + 1))
    for i in range(size):
        D[i, i] = -1
        D[i, i + 1] = 1
    return D / step


@dataclass(frozen=True)
class Grids:
    """Class initialised from number of grid cells to contain:

    grid cell width, center, edge and ghost grids and difference matrices
    """

    number_of_cells: int

    @cached_property
    def step(self) -> float:
        """Grid cell width"""
        return 1 / self.number_of_cells

    @cached_property
    def centers(self) -> NDArray:
        """Center grid"""
        return np.array(
            [-1 + (2 * i + 1) * self.step / 2 for i in range(self.number_of_cells)]
        )

    @cached_property
    def edges(self) -> NDArray:
        """Edge grid"""
        return np.array([-1 + i * self.step for i in range(self.number_of_cells + 1)])

    @cached_property
    def ghosts(self) -> NDArray:
        """Ghost grid"""
        return np.concatenate(
            (np.array([-1 - self.step / 2]), self.centers, np.array([self.step / 2]))
        )

    @cached_property
    def D_e(self) -> NDArray:
        """Difference matrix to differentiate edge grid quantities to the center grid"""
        return get_difference_matrix(self.number_of_cells, self.step)

    @cached_property
    def D_g(self) -> NDArray:
        """Difference matrix to differentiate ghost grid quantities to the edge grid"""
        return get_difference_matrix(self.number_of_cells + 1, self.step)


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


def average(points: NDArray) -> NDArray:
    """Returns arithmetic mean of adjacent points in an array

    takes ghosts -> edges -> centers
    """
    upper = points[1:]
    lower = points[:-1]
    return 0.5 * (upper + lower)


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


def calculate_ice_ocean_boundary_depth(liquid_fraction, edge_grid):
    r"""Calculate the depth of the ice ocean boundary as the edge position of the
    first cell from the bottom to be not completely liquid. I.e the first time the
    liquid fraction goes below 1.

    If the ice has made it to the bottom of the domain raise an error.

    If the domain is completely liquid set h=0.

    NOTE: depth is a positive quantity and our grid coordinate increases from -1 at the
    bottom of the domain to 0 at the top.

    :param liquid_fraction: liquid fraction on center grid
    :type liquid_fraction: Numpy Array (size I)
    :param edge_grid: The vertical coordinate positions of the edge grid.
    :type edge_grid: Numpy Array (size I+1)
    :return: positive depth value of ice ocean interface
    """
    # locate index on center grid where liquid fraction first drops below 1
    index = np.argmax(liquid_fraction < 1)

    # if domain is completely liquid set h=0
    if np.all(liquid_fraction == 1):
        index = edge_grid.size - 1

    # raise error if bottom of domain freezes
    if index == 0:
        raise ValueError("Ice ocean interface has reached bottom of domain")

    # ice interface is at bottom edge of first frozen cell
    depth = (-1) * edge_grid[index]
    return depth
