"""Celestine"""
import numpy as np
from celestine.params import Config, DarcyLawParams, ForcingConfig
from celestine.run_simulation import solve
from celestine.enthalpy_method import get_phase_masks, calculate_enthalpy_method
from celestine.velocities import calculate_velocities
import matplotlib.pyplot as plt
from celestine.grids import initialise_grids, get_difference_matrix
from celestine.logging_config import logger, log_time
from celestine.__init__ import __version__

logger.info(f"Celestine version {__version__}")


base = Config(
    name="base",
    total_time=4,
    savefreq=5e-2,
    darcy_law_params=DarcyLawParams(bubble_radius_scaled=0.1),
    forcing_config=ForcingConfig(temperature_forcing_choice="yearly"),
)
base.save()
status, duration = solve(base)
log_time(logger, duration, message="solve ran in ")


"""Analysis"""
with np.load("data/base.npz") as data:
    enthalpy = data["enthalpy"]
    salt = data["salt"]
    gas = data["gas"]
    pressure = data["pressure"]
    times = data["times"]
cfg = Config.load("data/base.yml")
phase_masks = get_phase_masks(enthalpy, salt, gas, cfg)
(
    temperature,
    liquid_fraction,
    gas_fraction,
    solid_fraction,
    liquid_salinity,
    dissolved_gas,
) = calculate_enthalpy_method(enthalpy, salt, gas, cfg, phase_masks)
D_g = get_difference_matrix(cfg.numerical_params.I + 1, cfg.numerical_params.step)
Vg, Wl, V = calculate_velocities(
    liquid_fraction, enthalpy, salt, gas, pressure, D_g, cfg
)
step, centers, edges, ghosts = initialise_grids(cfg.numerical_params.I)
for n, _ in enumerate(temperature[0, :]):
    plt.figure(figsize=(5, 5))
    plt.plot(
        gas_fraction[:, n],
        ghosts,
        "g*--",
    )
    plt.savefig(f"frames/gas_fraction/gas_fraction{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(
        salt[:, n],
        ghosts,
        "b*--",
    )
    plt.savefig(f"frames/salt/salt{n}.pdf")
    plt.close()
