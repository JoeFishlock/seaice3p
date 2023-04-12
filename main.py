import numpy as np
from params import Params
from lagged_solver import solve
from enthalpy_method import get_phase_masks, calculate_enthalpy_method
from velocities import calculate_velocities
import matplotlib.pyplot as plt
from grids import initialise_grids, get_difference_matrix, upwind

plt.style.use(["science", "grid"])
base = Params(
    name="base",
    far_gas_sat=1,
    total_time=4,
    temperature_forcing_choice="yearly",
    savefreq=1e-2,
)
# solve(base)
# print("Done solve")

mobile = Params(
    name="mobile",
    far_gas_sat=1,
    total_time=4,
    temperature_forcing_choice="yearly",
    savefreq=1e-2,
    bubble_radius_scaled=0.1,
)
# solve(mobile)
# print("Done solve")

res = Params(
    name="res",
    far_gas_sat=1,
    temperature_forcing_choice="yearly",
    savefreq=1e-2,
    bubble_radius_scaled=1.1,
    timestep=5e-5,
    I=100,
)
# solve(res)
# print("Done solve")

# profile = Params(
#     name="profile",
#     total_time=4,
#     temperature_forcing_choice="yearly",
#     constant_top_temperature=-1.5,
#     savefreq=1e-2,
#     bubble_radius_scaled=0.1,
#     far_gas_sat=1,
#     timestep=1.6e-4,
#     I=50,
# )
# solve(profile)
# print("Done simulation")

adapt = Params(
    name="adapt",
    total_time=8,
    temperature_forcing_choice="yearly",
    # constant_top_temperature=-1.5,
    savefreq=1e-1,
    bubble_radius_scaled=0.1,
    far_gas_sat=1,
    timestep=1.6e-4,
    I=50,
)
solve(adapt)
print("Done simulation")

"""Analysis"""
with np.load("data/adapt.npz") as data:
    enthalpy = data["enthalpy"]
    salt = data["salt"]
    gas = data["gas"]
    pressure = data["pressure"]
    times = data["times"]

phase_masks = get_phase_masks(enthalpy, salt, gas, adapt)
(
    temperature,
    liquid_fraction,
    gas_fraction,
    solid_fraction,
    liquid_salinity,
    dissolved_gas,
) = calculate_enthalpy_method(enthalpy, salt, gas, adapt, phase_masks)
D_g = get_difference_matrix(adapt.I + 1, adapt.step)
Vg, Wl, V = calculate_velocities(
    liquid_fraction, enthalpy, salt, gas, pressure, D_g, adapt
)
step, centers, edges, ghosts = initialise_grids(adapt.I)
for n, _ in enumerate(temperature[0, :]):
    plt.figure(figsize=(5, 5))
    plt.plot(
        temperature[:, n],
        ghosts,
        "r*--",
    )
    plt.savefig(f"frames_adapt/temperature{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(
        Wl[:, n],
        edges,
        "b*--",
    )
    plt.savefig(f"frames_adapt/Wl{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(
        upwind(gas_fraction[:, n], Vg[:, n]),
        edges,
        "g*--",
    )
    plt.savefig(f"frames_adapt/Wg{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(
        gas_fraction[:, n],
        ghosts,
        "g*--",
    )
    plt.savefig(f"frames_adapt/gas_fraction{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(
        solid_fraction[:, n],
        ghosts,
        "r*--",
    )
    plt.savefig(f"frames_adapt/solid_fraction{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(
        salt[:, n],
        ghosts,
        "b*--",
    )
    plt.savefig(f"frames_adapt/salt{n}.pdf")
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(
        gas[:, n],
        ghosts,
        "g*--",
    )
    plt.savefig(f"frames_adapt/gas{n}.pdf")
    plt.close()

# l, L, m, M, e, E, s, S = phase_masks
# phases = np.full_like(enthalpy, np.NaN)
# phases[l] = 0
# phases[L] = 1
# phases[m] = 2
# phases[M] = 3
# phases[e] = 4
# phases[E] = 5
# phases[s] = 6
# phases[S] = 7
# plt.figure()
# axes = plt.gca()
# axes.contourf(
#     times,
#     centers,
#     phases[1:-1, :],
#     levels=[-0.5, 0.5],
#     colors=[(0, 0, 1, 1)],
# )
# axes.contourf(
#     times,
#     centers,
#     phases[1:-1, :],
#     levels=[0.5, 1.5],
#     colors=[(0, 0, 1, 1)],
#     hatches=["\\\\\\"],
# )
# axes.contourf(
#     times,
#     centers,
#     phases[1:-1, :],
#     levels=[1.5, 2.5],
#     colors=[(0, 1, 1, 1)],
# )
# axes.contourf(
#     times,
#     centers,
#     phases[1:-1, :],
#     levels=[2.5, 3.5],
#     colors=[(0, 1, 1, 1)],
#     hatches=["\\\\\\"],
# )
# axes.contourf(
#     times,
#     centers,
#     phases[1:-1, :],
#     levels=[3.5, 4.5],
#     colors=[(1, 0, 0, 1)],
# )
# axes.contourf(
#     times,
#     centers,
#     phases[1:-1, :],
#     levels=[4.5, 5.5],
#     colors=[(1, 0, 0, 1)],
#     hatches=["\\\\\\"],
# )
# axes.contourf(
#     times,
#     centers,
#     phases[1:-1, :],
#     levels=[5.5, 6.5],
#     colors=[(0, 1, 0, 1)],
# )
# axes.contourf(
#     times,
#     centers,
#     phases[1:-1, :],
#     levels=[6.5, 7.5],
#     colors=[(0, 1, 0, 1)],
#     hatches=["\\\\\\"],
# )
# plt.savefig("phase_space.pdf")
