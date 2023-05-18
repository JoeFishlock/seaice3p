"""Celestine"""
import numpy as np
from celestine.params import Config, DarcyLawParams, ForcingConfig, NumericalParams
from celestine.run_simulation import solve
import matplotlib.pyplot as plt
from celestine.grids import initialise_grids, get_difference_matrix
from celestine.logging_config import logger, log_time
from celestine.__init__ import __version__
from celestine.state import State

logger.info(f"Celestine version {__version__}")


"""Generate one simulation config and save to data/base.yml
run the config and save data to data/base.npz
"""
base = Config(
    name="base",
    total_time=4,
    savefreq=5e-2,
    darcy_law_params=DarcyLawParams(bubble_radius_scaled=0.1),
    forcing_config=ForcingConfig(temperature_forcing_choice="yearly"),
    numerical_params=NumericalParams(solver="LU"),
)
base.save()
status, duration = solve(base)
log_time(logger, duration, message="solve ran in ")


"""Analysis load data from data/base.npz
plot
gas_fraction
salt
temperature
solid_fraction
save as frames in frames/gas_fraction etc...
"""
with np.load("data/base.npz") as data:
    enthalpy = data["enthalpy"]
    salt = data["salt"]
    gas = data["gas"]
    pressure = data["pressure"]
    times = data["times"]
cfg = Config.load("data/base.yml")

D_g = get_difference_matrix(cfg.numerical_params.I + 1, cfg.numerical_params.step)
step, centers, edges, ghosts = initialise_grids(cfg.numerical_params.I)

for n, time in enumerate(times):
    state = State(cfg, time, enthalpy[:, n], salt[:, n], gas[:, n], pressure[:, n])
    state.calculate_enthalpy_method()
    plt.figure(figsize=(5, 5))
    plt.plot(
        state.gas_fraction,
        centers,
        "g*--",
    )
    plt.savefig(f"frames/gas_fraction/gas_fraction{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(
        state.salt,
        centers,
        "b*--",
    )
    plt.savefig(f"frames/salt/salt{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(
        state.temperature,
        centers,
        "r*--",
    )
    plt.savefig(f"frames/temperature/temperature{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(
        state.solid_fraction,
        centers,
        "m*--",
    )
    plt.savefig(f"frames/solid_fraction/solid_fraction{n}.pdf")
    plt.close()
