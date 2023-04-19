"""Plot data in benchmarks/ 
plot gas fractions for each simulation at num equally spaced times to compare
"""
import numpy as np
from celestine.params import Config
from celestine.enthalpy_method import get_phase_masks, calculate_enthalpy_method
from celestine.velocities import calculate_velocities
import matplotlib.pyplot as plt
from celestine.grids import initialise_grids, get_difference_matrix
from celestine.__init__ import __version__
from glob import glob
from pathlib import Path

data_path = "benchmarks/"
num = 18
figs = []
for i in range(num):
    fig = plt.figure(figsize=(5, 5))
    figs.append(fig)

output_paths = glob(f"{data_path}*.npz")
for path in output_paths:
    """load in data and corresponding config"""
    with np.load(path) as data:
        enthalpy = data["enthalpy"]
        salt = data["salt"]
        gas = data["gas"]
        pressure = data["pressure"]
        times = data["times"]
    name = Path(path).stem
    cfg = Config.load(f"{data_path}{name}.yml")
    # get num evenly spaced indices
    idx = np.round(np.linspace(0, len(times) - 1, num)).astype(int)

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

    for i, fig in zip(idx, figs):
        plt.figure(fig.number)
        plt.plot(
            gas_fraction[:, i],
            ghosts,
            marker="o",
            fillstyle="none",
            label=f"{cfg.name}",
        )

for i, fig in zip(idx, figs):
    plt.figure(fig.number)
    plt.legend()
    plt.savefig(f"{data_path}gas_fraction_{times[i]:.2f}.pdf")
    plt.close()
