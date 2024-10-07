"""Script to run a simulation starting with dimensional parameters and plot output"""

from pathlib import Path
import matplotlib.pyplot as plt

from . import (
    __version__,
    solve,
    load_simulation,
    DimensionalParams,
    DimensionalBRW09Forcing,
    DimensionalMonoBubbleParams,
    DimensionalEQMGasParams,
    BRW09InitialConditions,
    DimensionalBRW09OceanForcing,
    NoBrineConvection,
    NumericalParams,
    get_config,
)

DATA_DIRECTORY = Path("example_data")
FRAMES_DIR = Path("example_data/frames")
SIMULATION_DIMENSIONAL_PARAMS = DimensionalParams(
    name="example",
    total_time_in_days=164,
    savefreq_in_days=3,
    lengthscale=2.4,
    gas_params=DimensionalEQMGasParams(),
    bubble_params=DimensionalMonoBubbleParams(bubble_radius=0.2e-3),
    numerical_params=NumericalParams(I=24),
    initial_conditions_config=BRW09InitialConditions(),
    forcing_config=DimensionalBRW09Forcing(),
    ocean_forcing_config=DimensionalBRW09OceanForcing(),
    brine_convection_params=NoBrineConvection(),
)


def create_and_save_config(
    data_directory: Path, simulation_dimensional_params: DimensionalParams
):
    data_directory.mkdir(exist_ok=True, parents=True)
    simulation_dimensional_params.save(data_directory)
    cfg = get_config(simulation_dimensional_params)
    cfg.save(data_directory)
    return cfg


def main(
    data_directory: Path,
    frames_directory: Path,
    simulation_dimensional_params: DimensionalParams,
):
    """Generate non dimensional simulation config and save along with dimensional
    config then run simulation and save data.
    """

    print(f"seaice3p version {__version__}")

    cfg = create_and_save_config(data_directory, simulation_dimensional_params)
    solve(cfg, data_directory, verbosity_level=1)

    # Analysis load simulation data
    # plot:
    # gas_fraction
    # salt
    # temperature
    # solid_fraction
    # save as frames in frames/gas_fraction etc...
    simulation_name = simulation_dimensional_params.name
    DIMENSIONAL_CONFIG_DATA_PATH = data_directory / f"{simulation_name}_dimensional.yml"
    results = load_simulation(
        DATA_DIRECTORY / "example_dimensional.yml",
        DATA_DIRECTORY / "example.npz",
        is_dimensional=True,
    )

    scales = results.cfg.scales
    dimensional_grid = scales.convert_to_dimensional_grid(results.grids.centers)
    dimensional_times = scales.convert_to_dimensional_time(results.times)

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

    for n, state in enumerate(results.states):
        plt.figure(figsize=(5, 5))
        plt.plot(
            state.gas_fraction,
            dimensional_grid,
            "g*--",
        )
        plt.title(f"{dimensional_times[n]:.0f} days")
        plt.xlabel("gas fraction")
        plt.ylabel("depth [m]")
        plt.savefig(GAS_FRACTION_DIR / f"gas_fraction{n}.pdf")
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.plot(
            scales.convert_to_dimensional_bulk_salinity(state.salt),
            dimensional_grid,
            "b*--",
        )
        plt.title(f"{dimensional_times[n]:.0f} days")
        plt.xlabel("bulk salinity [g/kg]")
        plt.ylabel("depth [m]")
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
        plt.title(f"{dimensional_times[n]:.0f} days")
        plt.xlabel("temperature [deg C]")
        plt.ylabel("depth [m]")
        plt.savefig(TEMPERATURE_DIR / f"temperature{n}.pdf")
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.plot(
            state.solid_fraction,
            dimensional_grid,
            "m*--",
        )
        plt.title(f"{dimensional_times[n]:.0f} days")
        plt.xlabel("solid fraction")
        plt.ylabel("depth [m]")
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
        plt.title(f"{dimensional_times[n]:.0f} days")
        plt.xlabel("bulk argon [micromole/L]")
        plt.ylabel("depth [m]")
        plt.savefig(BULK_AIR_DIR / f"bulk_air{n}.pdf")
        plt.close()

    for converter, unit, attr in zip(
        [
            scales.convert_to_dimensional_temperature,
            scales.convert_to_dimensional_bulk_salinity,
            lambda x: x,
            lambda x: x,
            scales.convert_to_dimensional_bulk_gas,
        ],
        ["[deg C]", "[g/kg]", "", "", "[kg/m3]"],
        ["temperature", "salt", "gas_fraction", "solid_fraction", "bulk_gas"],
    ):
        plt.figure()
        plt.contourf(
            dimensional_times, dimensional_grid, converter(getattr(results, attr))
        )
        plt.colorbar()
        plt.title(f"{attr} {unit}")
        plt.xlabel("time [days]")
        plt.ylabel("depth [m]")
        plt.savefig(data_directory / f"contours_{attr}.pdf")


if __name__ == "__main__":
    main(DATA_DIRECTORY, FRAMES_DIR, SIMULATION_DIMENSIONAL_PARAMS)
