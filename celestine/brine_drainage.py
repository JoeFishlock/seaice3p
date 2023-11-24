"""Module to calculate the Rees Jones and Worster 2014
parameterisation for brine convection velocity and the corresponding
fluxes of heat, salt, dissolved and free phase gas"""

import numpy as np
import matplotlib.pyplot as plt


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
        print("fired")
        index = edge_grid.size - 1

    # raise error if bottom of domain freezes
    if index == 0:
        raise ValueError("Ice ocean interface has reached bottom of domain")

    # ice interface is at bottom edge of first frozen cell
    depth = (-1) * edge_grid[index]
    return depth


if __name__ == "__main__":
    I = 20
    liquid_fraction = [0.8] * int(I / 2) + [0.2] * int(I / 2)
    liquid_fraction = np.array(liquid_fraction)

    # liquid_fraction = np.linspace(1, 0.8, I)

    edge_grid = np.linspace(-1, 0, I + 1)
    first_center = 0.5 * (edge_grid[0] + edge_grid[1])
    last_center = 0.5 * (edge_grid[-1] + edge_grid[-2])
    center_grid = np.linspace(first_center, last_center, I)
    h = calculate_ice_ocean_boundary_depth(liquid_fraction, edge_grid)
    plt.figure()
    plt.plot(liquid_fraction, center_grid, "b*--", label="liquid fraction")
    plt.axhline(-h, label="ice depth")
    plt.legend()
    plt.show()
