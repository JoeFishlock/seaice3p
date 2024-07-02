"""Module providing functions to initialise the different grids and interpolate
quantities between them.
"""

from dataclasses import dataclass
from functools import cached_property
import numpy as np
from numpy.typing import NDArray


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
