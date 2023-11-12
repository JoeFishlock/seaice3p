import numpy as np
import matplotlib.pyplot as plt
import celestine.velocities as vel
from celestine.grids import get_difference_matrix, geometric
from celestine.params import Config, DarcyLawParams


class MockStateBCs:
    def __init__(self, liquid_fraction):
        self.liquid_fraction = liquid_fraction
        self.pressure = np.zeros_like(liquid_fraction)


mono = Config(
    "mono-disperse",
    darcy_law_params=DarcyLawParams(
        B=2e7,
        bubble_radius_scaled=0.5,
        pore_throat_scaling=0.5,
        drag_exponent=2.5,
        liquid_velocity=0.0,
    ),
)

power_law = Config(
    "poly-disperse",
    darcy_law_params=DarcyLawParams(
        B=2e7,
        bubble_size_distribution_type="power_law",
        pore_throat_scaling=0.5,
        drag_exponent=2.5,
        liquid_velocity=0.0,
        bubble_distribution_power=1.5,
        maximum_bubble_radius_scaled=1,
        minimum_bubble_radius_scaled=1e-3,
    ),
)
liquid_fraction = np.linspace(0, 1, mono.numerical_params.I + 2)
D_g = get_difference_matrix(mono.numerical_params.I + 1, mono.numerical_params.step)
mono_vel, _, _ = vel.calculate_velocities(
    state_BCs=MockStateBCs(liquid_fraction=liquid_fraction),
    cfg=mono,
)
poly_vel, _, _ = vel.calculate_velocities(
    state_BCs=MockStateBCs(liquid_fraction=liquid_fraction),
    cfg=power_law,
)
plt.figure()
plt.plot(geometric(liquid_fraction), mono_vel, "r*--")
plt.plot(geometric(liquid_fraction), poly_vel, "b*--")
plt.show()
