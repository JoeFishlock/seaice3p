"""script to visualise simulation data

usage:
python -m seaice3p.plot "glob pattern to find npz files" Optional[True/False]

assumes the simulation configurations are to be found in the same directory as the data.
If the simulation is a non-dimensional configuration file add the False option after
the glob pattern.
"""
import sys
from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from . import load_simulation


def plot(
    config_path: Path,
    data_path: Path,
    is_dimensional: bool = True,
    output_dir: Path = Path("."),
    animation_length: float = 10,
) -> None:
    results = load_simulation(
        config_path,
        data_path,
        is_dimensional=is_dimensional,
    )

    scales = results.cfg.scales
    dimensional_grid = scales.convert_to_dimensional_grid(results.grids.centers)
    dimensional_times = scales.convert_to_dimensional_time(results.times)
    save_dir = output_dir / results.cfg.name
    save_dir.mkdir(exist_ok=True, parents=True)

    for converter, unit, attr in zip(
        [
            scales.convert_to_dimensional_temperature,
            scales.convert_to_dimensional_bulk_salinity,
            lambda x: x,
            lambda x: x,
            scales.convert_to_dimensional_bulk_gas,
            lambda x: x,
            lambda x: x,
        ],
        ["[deg C]", "[g/kg]", "", "", "[kg/m3]", "[ng/g]", "[micromole/L]"],
        [
            "temperature",
            "salt",
            "gas_fraction",
            "solid_fraction",
            "bulk_gas",
            "oil_mass_ratio",
            "bulk_argon",
        ],
    ):
        data = converter(getattr(results, attr))
        plt.figure()
        if attr == "solid_fraction":
            cmap = plt.colormaps["Blues_r"].with_extremes(under="b", over="w")
            plt.contour(
                dimensional_times, dimensional_grid, data, [0.0], color="r", zorder=100
            )
        elif attr == "temperature":
            cmap = plt.colormaps["OrRd"]
        else:
            cmap = plt.colormaps["viridis"]

        plt.contourf(dimensional_times, dimensional_grid, data, cmap=cmap)
        plt.colorbar()
        plt.title(f"{attr} {unit}")
        plt.xlabel("time [days]")
        plt.ylabel("depth [m]")
        plt.savefig(save_dir / f"contours_{attr}.pdf")

        fig, ax = plt.subplots()
        line = ax.plot(data[:, 0], dimensional_grid, "k.-")[0]
        max = np.max(np.max(data))
        min = np.min(np.min(data))
        width = max - min
        ax.set(
            xlim=[min - 0.01 * width, max + 0.01 * width],
            ylim=[dimensional_grid[0] - 0.1, dimensional_grid[-1] + 0.1],
            xlabel=unit,
            ylabel="z [m]",
            title=f"{results.cfg.name}: {attr}",
        )

        def update(frame):
            # for each frame, update the data stored on each artist.
            # update the line plot:
            line.set_xdata(data[:, frame])
            line.set_ydata(dimensional_grid)
            ax.set(
                title=f"{results.cfg.name}: {attr} at {dimensional_times[frame]:.1f} days"
            )
            return (line,)

        ani = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=len(dimensional_times),
            interval=animation_length * 1000 / dimensional_times.size,
        )
        ani.save(filename=save_dir / f"{attr}.mp4", writer="ffmpeg")
        plt.close()


if __name__ == "__main__":
    """Command line arguments:
    output directory path: specify path for output
    config directory path: specify path of directory containing configurations
    data directory glob: specify a glob pattern to find path to all data files
    is_dimensional: specify if simulation configurations are dimensional (boolean)
    """
    data_glob = glob(sys.argv[1])
    data_paths = [Path(path) for path in data_glob]
    if len(sys.argv) <= 2:
        is_dimensional = True
    else:
        is_dimensional = sys.argv[2]

    for data_path in data_paths:
        name = data_path.stem
        if is_dimensional:
            config_path = data_path.parent / f"{name}_dimensional.yml"
        else:
            config_path = data_path.parent / f"{name}.yml"
        plot(
            config_path,
            data_path,
            is_dimensional=is_dimensional,
            output_dir=data_path.parent,
        )
