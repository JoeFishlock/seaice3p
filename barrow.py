"""Script to run a simulation starting with dimensional parameters"""

import numpy as np
from celestine.params import Config, ForcingConfig, NumericalParams
from celestine.dimensional_params import DimensionalParams
from celestine.run_simulation import solve
import matplotlib.pyplot as plt
from celestine.logging_config import logger, log_time
from celestine.__init__ import __version__
from celestine.state import State

logger.info(f"Celestine version {__version__}")


"""Generate one simulation config and save to data/base.yml
run the config and save data to data/base.npz
"""
barrow_dimensional_params = DimensionalParams(name="barrow", bubble_radius=1e-3)
barrow_dimensional_params.save()
barrow = barrow_dimensional_params.get_config(
    forcing_config=ForcingConfig(
        temperature_forcing_choice="yearly", period=barrow_dimensional_params.total_time
    ),
    numerical_params=NumericalParams(solver="SCI", I=50),
)
barrow.save()
status, duration = solve(barrow)
log_time(logger, duration, message="solve ran in ")


"""Analysis load data from data/base.npz
plot
gas_fraction
salt
temperature
solid_fraction
save as frames in frames/gas_fraction etc...
"""
with np.load("data/barrow.npz") as data:
    enthalpy = data["enthalpy"]
    salt = data["salt"]
    gas = data["gas"]
    pressure = data["pressure"]
    times = data["times"]
cfg = Config.load("data/barrow.yml")
scales = DimensionalParams.load("data/barrow_dimensional.yml").get_scales()

for n, time in enumerate(times):
    state = State(cfg, time, enthalpy[:, n], salt[:, n], gas[:, n], pressure[:, n])
    state.calculate_enthalpy_method()
    plt.figure(figsize=(5, 5))
    plt.plot(
        state.gas_fraction,
        state.grid,
        "g*--",
    )
    plt.savefig(f"frames/gas_fraction/gas_fraction{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    dimensional_temperature = scales.convert_to_dimensional_temperature(
        state.temperature
    )
    dimensional_grid = scales.convert_to_dimensional_grid(state.grid)
    plt.plot(
        dimensional_temperature,
        dimensional_grid,
        "r*--",
    )
    plt.savefig(f"frames/temperature/temperature{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(
        state.solid_fraction,
        state.grid,
        "m*--",
    )
    plt.savefig(f"frames/solid_fraction/solid_fraction{n}.pdf")
    plt.close()
