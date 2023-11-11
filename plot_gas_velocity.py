import numpy as np
import matplotlib.pyplot as plt
import celestine.velocities as vel
from celestine.grids import get_difference_matrix, geometric
from celestine.params import Config, DarcyLawParams


class MockStateBCs:
    def __init__(self, liquid_fraction):
        self.liquid_fraction = liquid_fraction
        self.pressure = np.zeros_like(liquid_fraction)


cfg = Config(
    "base",
    darcy_law_params=DarcyLawParams(
        B=2e7,
        bubble_radius_scaled=0.5,
        pore_throat_scaling=0.5,
        drag_exponent=2.5,
        liquid_velocity=0.0,
    ),
)
liquid_fraction = np.linspace(0, 1, cfg.numerical_params.I + 2)
D_g = get_difference_matrix(cfg.numerical_params.I + 1, cfg.numerical_params.step)
Vg, Wl, V = vel.calculate_velocities(
    state_BCs=MockStateBCs(liquid_fraction=liquid_fraction),
    D_g=D_g,
    cfg=cfg,
)
plt.figure()
plt.plot(geometric(liquid_fraction), Vg, "r*--")
plt.show()
