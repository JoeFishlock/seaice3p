"""Script to run a simulation starting with dimensional parameters"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from celestine.logging_config import logger, log_time
from celestine.__init__ import __version__
from celestine.params import (
    Config,
)
from celestine.dimensional_params import DimensionalParams
from celestine.run_simulation import solve
from celestine.state import State


def create_and_save_config(data_directory: Path, simulation_dimensional_params: dict):
    data_directory.mkdir(exist_ok=True, parents=True)
    dimensional_params = DimensionalParams(**simulation_dimensional_params)
    dimensional_params.save(data_directory)
    cfg = dimensional_params.get_config()
    cfg.save(data_directory)
    return cfg


def main(
    data_directory: Path, frames_directory: Path, simulation_dimensional_params: dict
):
    """Generate non dimensional simulation config and save along with dimensional
    config then run simulation and save data.
    """

    logger.info(f"Celestine version {__version__}")

    cfg = create_and_save_config(data_directory, simulation_dimensional_params)
    _, duration = solve(cfg, data_directory)
    log_time(logger, duration, message="solve ran in ")

    """Analysis load simulation data
    plot:
    gas_fraction
    salt
    temperature
    solid_fraction
    save as frames in frames/gas_fraction etc...
    """
    simulation_name = simulation_dimensional_params["name"]
    SIMULATION_DATA_PATH = data_directory / f"{simulation_name}.npz"
    CONFIG_DATA_PATH = data_directory / f"{simulation_name}.yml"
    DIMENSIONAL_CONFIG_DATA_PATH = data_directory / f"{simulation_name}_dimensional.yml"

    with np.load(SIMULATION_DATA_PATH) as data:
        enthalpy = data["enthalpy"]
        salt = data["salt"]
        gas = data["gas"]
        pressure = data["pressure"]
        times = data["times"]
    cfg = Config.load(CONFIG_DATA_PATH)
    scales = DimensionalParams.load(DIMENSIONAL_CONFIG_DATA_PATH).get_scales()

    GAS_FRACTION_DIR = frames_directory / "gas_fraction/"
    GAS_FRACTION_DIR.mkdir(exist_ok=True, parents=True)

    TEMPERATURE_DIR = frames_directory / "temperature/"
    TEMPERATURE_DIR.mkdir(exist_ok=True, parents=True)

    SOLID_FRAC_DIR = frames_directory / "solid_fraction/"
    SOLID_FRAC_DIR.mkdir(exist_ok=True, parents=True)

    BULK_AIR_DIR = frames_directory / "bulk_air/"
    BULK_AIR_DIR.mkdir(exist_ok=True, parents=True)

    BULK_SALT_DIR = frames_directory / "bulk_salt/"
    BULK_SALT_DIR.mkdir(exist_ok=True, parents=True)

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
        plt.savefig(GAS_FRACTION_DIR / f"gas_fraction{n}.pdf")
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.plot(
            state.salt,
            dimensional_grid,
            "b*--",
        )
        plt.savefig(BULK_SALT_DIR / f"bulk_salt{n}.pdf")
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
        plt.savefig(TEMPERATURE_DIR / f"temperature{n}.pdf")
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.plot(
            state.solid_fraction,
            dimensional_grid,
            "m*--",
        )
        plt.savefig(SOLID_FRAC_DIR / f"solid_fraction{n}.pdf")
        plt.close()

        plt.figure(figsize=(5, 5))
        dimensional_bulk_air = scales.convert_to_dimensional_bulk_gas(state.gas)
        argon_micromole_per_liter = (
            scales.convert_dimensional_bulk_air_to_argon_content(dimensional_bulk_air)
        )
        plt.plot(
            argon_micromole_per_liter,
            dimensional_grid,
            "m*--",
        )
        plt.savefig(BULK_AIR_DIR / f"bulk_air{n}.pdf")
        plt.close()


if __name__ == "__main__":
    DATA_DIRECTORY = Path("data")
    FRAMES_DIR = Path("frames")
    SIMULATION_DIMENSIONAL_PARAMS = {
        "name": "barrow",
        "total_time_in_days": 164,
        "savefreq_in_days": 3,
        "bubble_radius": 0.2e-3,
        "lengthscale": 2.4,
        "solver": "SCI",
        "I": 24,
        "temperature_forcing_choice": "barrow_2009",
        "initial_conditions_choice": "barrow_2009",
    }
    main(DATA_DIRECTORY, FRAMES_DIR, SIMULATION_DIMENSIONAL_PARAMS)
