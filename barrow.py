"""Script to run a simulation starting with dimensional parameters"""

import numpy as np
from pathlib import Path
from celestine.params import (
    Config,
    ForcingConfig,
    NumericalParams,
    BoundaryConditionsConfig,
)
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
DATA_DIRECTORY = Path("data/")
barrow_dimensional_params = DimensionalParams(
    name="barrow",
    total_time_in_days=164,
    savefreq_in_days=3,
    bubble_radius=0.2e-3,
    lengthscale=2.4,
)
barrow_dimensional_params.save(DATA_DIRECTORY)
barrow = barrow_dimensional_params.get_config(
    forcing_config=ForcingConfig(temperature_forcing_choice="barrow_2009"),
    numerical_params=NumericalParams(solver="SCI", I=24),
    boundary_conditions_config=BoundaryConditionsConfig(
        initial_conditions_choice="barrow_2009"
    ),
)
barrow.save(DATA_DIRECTORY)
status, duration = solve(barrow, DATA_DIRECTORY)
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
    dimensional_grid = scales.convert_to_dimensional_grid(state.grid)

    plt.figure(figsize=(5, 5))
    plt.plot(
        state.gas_fraction,
        dimensional_grid,
        "g*--",
    )
    plt.savefig(f"frames/gas_fraction/gas_fraction{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    dimensional_temperature = scales.convert_to_dimensional_temperature(
        state.temperature
    )
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
        dimensional_grid,
        "m*--",
    )
    plt.savefig(f"frames/solid_fraction/solid_fraction{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    dimensional_bulk_air = scales.convert_to_dimensional_bulk_gas(state.gas)
    argon_micromole_per_liter = scales.convert_dimensional_bulk_air_to_argon_content(
        dimensional_bulk_air
    )
    plt.plot(
        argon_micromole_per_liter,
        dimensional_grid,
        "m*--",
    )
    plt.savefig(f"frames/bulk_air/bulk_air{n}.pdf")
    plt.close()
