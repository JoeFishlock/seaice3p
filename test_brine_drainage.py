import numpy as np
import matplotlib.pyplot as plt
from celestine.brine_drainage import (
    calculate_ice_ocean_boundary_depth,
    calculate_integrated_mean_permeability,
    calculate_Rayleigh,
    get_convecting_region_height,
    get_effective_Rayleigh_number,
    calculate_brine_channel_strength,
    calculate_brine_convection_liquid_velocity,
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

"""Plot Rayleigh Number with Depth"""
liquid_fraction = [1] * int(I / 2) + list(np.linspace(1, 0, int(I / 2)))
liquid_fraction = np.array(liquid_fraction)
liquid_salinity = [0] * int(I / 2) + list(np.linspace(0, 1, int(I / 2)))
liquid_salinity = np.array(liquid_salinity)
Rayleigh = calculate_Rayleigh(
    center_grid, edge_grid, liquid_salinity, liquid_fraction, cfg
)
h = calculate_ice_ocean_boundary_depth(liquid_fraction, edge_grid)
top_boundary = get_convecting_region_height(Rayleigh, edge_grid, cfg)
print("effective Rayleigh", get_effective_Rayleigh_number(Rayleigh, cfg))
print(
    "brine channel strenght",
    calculate_brine_channel_strength(Rayleigh, h, top_boundary, cfg),
)
print("ice depth", h)
plt.figure()
plt.plot(Rayleigh, center_grid, "r*--")
plt.axhline(top_boundary)
plt.xlabel("Rayleigh Number")
plt.ylabel("depth")


"""Plot the liquid velocity"""
Wl = calculate_brine_convection_liquid_velocity(
    liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg
)
plt.figure()
plt.plot(Wl, edge_grid, "m*--")
plt.xlabel("Liquid Darcy Velocity")
plt.ylabel("Depth")
plt.show()
