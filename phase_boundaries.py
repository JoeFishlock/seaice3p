import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from params import Params

plt.style.use(["science", "ieee", "grid"])


def calculate_liquidus(salt, gas, params):
    liquidus = np.full_like(salt, np.NaN)
    chi = params.expansion_coefficient
    C = params.concentration_ratio
    is_sub = gas <= chi
    is_super = ~is_sub
    liquidus[is_sub] = -salt[is_sub]
    liquidus[is_super] = -(salt[is_super] + C * ((gas[is_super] - chi) / (1 - chi)))
    return liquidus


def calculate_eutectic(salt, gas, params):
    eutectic = np.full_like(salt, np.NaN)
    chi = params.expansion_coefficient
    C = params.concentration_ratio
    St = params.stefan_number
    eutectic_liquid_fraction = (salt + C) / (1 + C)
    is_sub = gas <= chi * eutectic_liquid_fraction
    is_super = ~is_sub
    eutectic[is_sub] = -1 - St * (1 - eutectic_liquid_fraction[is_sub])
    eutectic[is_super] = (
        -(1 - gas[is_super] + chi * eutectic_liquid_fraction[is_super])
        - (1 - gas[is_super] + eutectic_liquid_fraction[is_super] * (chi - 1)) * St
    )
    return eutectic


def calculate_solidus(salt, gas, params):
    solidus = np.full_like(salt, np.NaN)
    St = params.stefan_number
    is_sub = gas <= 0
    is_super = ~is_sub
    solidus[is_sub] = -1 - St
    solidus[is_super] = (1 - gas[is_super]) * (-1 - St)
    return solidus


def calculate_saturation(enthalpy, salt, params):
    chi = params.expansion_coefficient
    St = params.stefan_number
    C = params.concentration_ratio
    saturation = np.full_like(enthalpy, np.NaN)
    no_gas = np.zeros_like(salt)
    liquidus = calculate_liquidus(salt, no_gas, params)
    eutectic = calculate_eutectic(salt, no_gas, params)
    solidus = calculate_solidus(salt, no_gas, params)
    is_liquid = enthalpy >= liquidus
    is_mush = (enthalpy >= eutectic) & (enthalpy < liquidus)
    is_eutectic = (enthalpy >= solidus) & (enthalpy < eutectic)
    is_solid = enthalpy < solidus
    saturation[is_liquid] = chi
    B = enthalpy[is_mush] + C + St
    A = C * enthalpy[is_mush] - St * salt[is_mush]
    mush_temperature = (1 / 2) * (B - np.sqrt(B**2 - 4 * A))
    saturation[is_mush] = chi * (1 - ((mush_temperature - enthalpy[is_mush]) / St))
    saturation[is_eutectic] = chi * (1 + (enthalpy[is_eutectic] + 1) / St)
    saturation[is_solid] = 0
    return saturation


def calculate_max_salt(gas, params):
    max_salt = np.full_like(gas, np.NaN)
    chi = params.expansion_coefficient
    C = params.concentration_ratio
    is_sub = gas <= chi
    is_super = ~is_sub
    max_salt[is_sub] = 1
    max_salt[is_super] = 1 - (gas[is_super] - chi) / (1 - chi) * (1 + C)
    return max_salt


def get_phase_masks(enthalpy, salt, gas, params):
    liquidus = calculate_liquidus(salt, gas, params)
    eutectic = calculate_eutectic(salt, gas, params)
    solidus = calculate_solidus(salt, gas, params)
    saturation = calculate_saturation(enthalpy, salt, params)
    is_liquid = enthalpy >= liquidus
    is_mush = (enthalpy >= eutectic) & (enthalpy < liquidus)
    is_eutectic = (enthalpy >= solidus) & (enthalpy < eutectic)
    is_solid = enthalpy < solidus
    is_sub = gas <= saturation
    is_super = ~is_sub
    l = is_liquid & is_sub
    L = is_liquid & is_super
    m = is_mush & is_sub
    M = is_mush & is_super
    e = is_eutectic & is_sub
    E = is_eutectic & is_super
    s = is_solid & is_sub
    S = is_solid & is_super
    return l, L, m, M, e, E, s, S


def plot_phases(gas_value, params, axes):
    max_salt = calculate_max_salt(np.array([gas_value]), params)
    max_salt = max_salt[0]
    salt = np.linspace(-params.concentration_ratio, max_salt, 1000)
    enthalpy = np.linspace(-6, 1, 1000)
    salt_mesh, enthalpy_mesh = np.meshgrid(salt, enthalpy)
    gas = np.full_like(salt_mesh, gas_value)
    l, L, m, M, e, E, s, S = get_phase_masks(enthalpy_mesh, salt_mesh, gas, params)
    phases = np.full_like(salt_mesh, np.NaN)
    phases[l] = 0
    phases[L] = 1
    phases[m] = 2
    phases[M] = 3
    phases[e] = 4
    phases[E] = 5
    phases[s] = 6
    phases[S] = 7
    axes.contourf(
        salt_mesh,
        enthalpy_mesh,
        phases,
        levels=[-0.5, 0.5],
        colors=[(0, 0, 1, 1)],
    )
    axes.contourf(
        salt_mesh,
        enthalpy_mesh,
        phases,
        levels=[0.5, 1.5],
        colors=[(0, 0, 1, 1)],
        hatches=["\\\\\\"],
    )
    axes.contourf(
        salt_mesh,
        enthalpy_mesh,
        phases,
        levels=[1.5, 2.5],
        colors=[(0, 1, 1, 1)],
    )
    axes.contourf(
        salt_mesh,
        enthalpy_mesh,
        phases,
        levels=[2.5, 3.5],
        colors=[(0, 1, 1, 1)],
        hatches=["\\\\\\"],
    )
    axes.contourf(
        salt_mesh,
        enthalpy_mesh,
        phases,
        levels=[3.5, 4.5],
        colors=[(1, 0, 0, 1)],
    )
    axes.contourf(
        salt_mesh,
        enthalpy_mesh,
        phases,
        levels=[4.5, 5.5],
        colors=[(1, 0, 0, 1)],
        hatches=["\\\\\\"],
    )
    axes.contourf(
        salt_mesh,
        enthalpy_mesh,
        phases,
        levels=[5.5, 6.5],
        colors=[(0, 1, 0, 1)],
    )
    axes.contourf(
        salt_mesh,
        enthalpy_mesh,
        phases,
        levels=[6.5, 7.5],
        colors=[(0, 1, 0, 1)],
        hatches=["\\\\\\"],
    )


def plot_phase_boundaries(gas_value, params, axes):
    max_salt = calculate_max_salt(np.array([gas_value]), params)
    max_salt = max_salt[0]
    salt = np.linspace(-params.concentration_ratio, max_salt, 1000)
    gas = np.full_like(salt, gas_value)
    liquidus = calculate_liquidus(salt, gas, params)
    eutectic = calculate_eutectic(salt, gas, params)
    solidus = calculate_solidus(salt, gas, params)
    axes.plot(salt, liquidus, "b")
    axes.plot(salt, eutectic, "r")
    axes.plot(salt, solidus, "g")


def plot_saturation_contour(gas_value, params, axes):
    H_range = np.linspace(-5, 1, 1000)
    max_salt = calculate_max_salt(np.array([gas_value]), params)
    max_salt = max_salt[0]
    salt = np.linspace(-params.concentration_ratio, max_salt, 1000)
    S, H = np.meshgrid(salt, H_range)
    saturation = calculate_saturation(H, S, params)
    axes.contour(
        S,
        H,
        saturation,
        levels=[0, gas_value],
        colors="k",
    )


if __name__ == "__main__":
    params = Params(name="base")
    chi = params.expansion_coefficient

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(5, 7))
    axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]
    for ax, gas_value in zip(axes, [0, chi * 0.25, chi * 0.5, chi * 0.75, chi, 0.3]):
        ax.set_xlim(-params.concentration_ratio, 1)
        plot_phases(gas_value, params, ax)
        plot_phase_boundaries(gas_value, params, ax)
        try:
            plot_saturation_contour(gas_value, params, ax)
        except:
            pass
        ax.set_title(f"bulk gas {gas_value:.3f}")
    plt.savefig("phases.pdf")
