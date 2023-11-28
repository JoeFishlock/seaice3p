import numpy as np
import matplotlib.pyplot as plt
from celestine.brine_drainage import (
    calculate_ice_ocean_boundary_depth,
    calculate_integrated_mean_permeability,
)
from celestine.params import Config, NumericalParams, DarcyLawParams

"""Plot showing location of ice ocean interface"""
I = 20
liquid_fraction = [1] * int(I / 2) + [0.2] * int(I / 2)
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

"""Print values of average permeability in ice"""
cfg = Config(
    "test",
    numerical_params=NumericalParams(I=200),
    darcy_law_params=DarcyLawParams(
        porosity_threshold=False, porosity_threshold_value=0.024
    ),
)
I = cfg.numerical_params.I
liquid_fraction = [1] * int(I / 2) + [0.2] * int(I / 2)
liquid_fraction = np.array(liquid_fraction)
edge_grid = np.linspace(-1, 0, I + 1)
first_center = 0.5 * (edge_grid[0] + edge_grid[1])
last_center = 0.5 * (edge_grid[-1] + edge_grid[-2])
center_grid = np.linspace(first_center, last_center, I)
h = calculate_ice_ocean_boundary_depth(liquid_fraction, edge_grid)
print("ice depth", h)
print("edges", edge_grid)
print("centers", center_grid)
integrated_perm = np.array(
    [
        calculate_integrated_mean_permeability(
            z=center,
            liquid_fraction=liquid_fraction,
            ice_depth=h,
            cell_centers=center_grid,
            cfg=cfg,
        )
        for center in center_grid
    ]
)
print("integrated permeability", integrated_perm)
