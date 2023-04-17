"""Celestine version 0.2.0"""
import numpy as np
from params import Params
from lax_friedrich_solver import solve
from enthalpy_method import get_phase_masks, calculate_enthalpy_method
from velocities import calculate_velocities
import matplotlib.pyplot as plt
from grids import initialise_grids, get_difference_matrix
from logging_config import logger, log_time

logger.info("Celestine version 0.2.0")

base = Params(
    name="base",
    far_gas_sat=1,
    total_time=4,
    bubble_radius_scaled=0.1,
    temperature_forcing_choice="yearly",
    savefreq=5e-2,
)

status, duration = solve(base)
log_time(logger, duration, message="solve ran in ")


"""Analysis"""
with np.load("data/base.npz") as data:
    enthalpy = data["enthalpy"]
    salt = data["salt"]
    gas = data["gas"]
    pressure = data["pressure"]
    times = data["times"]

phase_masks = get_phase_masks(enthalpy, salt, gas, base)
(
    temperature,
    liquid_fraction,
    gas_fraction,
    solid_fraction,
    liquid_salinity,
    dissolved_gas,
) = calculate_enthalpy_method(enthalpy, salt, gas, base, phase_masks)
D_g = get_difference_matrix(base.I + 1, base.step)
Vg, Wl, V = calculate_velocities(
    liquid_fraction, enthalpy, salt, gas, pressure, D_g, base
)
step, centers, edges, ghosts = initialise_grids(base.I)
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
