from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ..equations.RJW14.brine_drainage import (
    calculate_integrated_mean_permeability,
    calculate_Rayleigh,
    get_convecting_region_height,
    get_effective_Rayleigh_number,
    calculate_brine_channel_strength,
    calculate_brine_convection_liquid_velocity,
    calculate_brine_channel_sink,
)
from ..params import (
    Config,
    NumericalParams,
    MonoBubbleParams,
    RJW14Params,
    EQMPhysicalParams,
    ConstantForcing,
    UniformInitialConditions,
    FixedTempOceanForcing,
)
from ..grids import calculate_ice_ocean_boundary_depth


def main(output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    OUTPUT_FILE = output_dir / "diagnostics.txt"

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
    plt.savefig(output_dir / "liquid_fraction.pdf")
    plt.close()

    """Print values of average permeability in ice"""
    cfg = Config(
        name="test",
        total_time=4,
        savefreq=0.1,
        numerical_params=NumericalParams(I=200),
        brine_convection_params=RJW14Params(),
        bubble_params=MonoBubbleParams(),
        physical_params=EQMPhysicalParams(),
        forcing_config=ConstantForcing(),
        ocean_forcing_config=FixedTempOceanForcing(),
        initial_conditions_config=UniformInitialConditions(),
    )
    I = cfg.numerical_params.I
    liquid_fraction = [1] * int(I / 2) + [0.2] * int(I / 2)
    liquid_fraction = np.array(liquid_fraction)
    edge_grid = np.linspace(-1, 0, I + 1)
    first_center = 0.5 * (edge_grid[0] + edge_grid[1])
    last_center = 0.5 * (edge_grid[-1] + edge_grid[-2])
    center_grid = np.linspace(first_center, last_center, I)
    h = calculate_ice_ocean_boundary_depth(liquid_fraction, edge_grid)
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
    with open(OUTPUT_FILE, "w") as text_file:
        text_file.write(f"ice depth {h}\n")
        text_file.write(f"edges {edge_grid}\n")
        text_file.write(f"centers {center_grid}\n")
        text_file.write(f"integrated permeability {integrated_perm}")

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

    with open(OUTPUT_FILE, "a") as text_file:
        text_file.write(
            f"effective Rayleigh {get_effective_Rayleigh_number(Rayleigh, cfg)}\n"
        )
        text_file.write(
            f"brine channel strength {calculate_brine_channel_strength(Rayleigh, h, top_boundary, cfg)}\n"
        )
        text_file.write(f"ice depth {h}\n")

    plt.figure()
    plt.plot(Rayleigh, center_grid, "r*--")
    plt.axhline(top_boundary)
    plt.xlabel("Rayleigh Number")
    plt.ylabel("depth")
    plt.savefig(output_dir / "Rayleigh_number.pdf")
    plt.close()

    """Plot the liquid velocity"""
    Wl = calculate_brine_convection_liquid_velocity(
        liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg
    )
    plt.figure()
    plt.plot(Wl, edge_grid, "m*--")
    plt.xlabel("Liquid Darcy Velocity")
    plt.ylabel("Depth")
    plt.savefig(output_dir / "liquid_Darcy_velocity.pdf")
    plt.close()

    """Plot the sink term"""
    sink = calculate_brine_channel_sink(
        liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg
    )
    plt.figure()
    plt.plot(sink, center_grid, "g*--")
    plt.xlabel("Brine Channel Sink Strength")
    plt.ylabel("Depth")
    plt.savefig(output_dir / "sink_term.pdf")
    plt.close()


if __name__ == "__main__":
    OUTPUT_DIR = Path("brine_drainage_parameterisation_diagnostics")
    main(OUTPUT_DIR)
