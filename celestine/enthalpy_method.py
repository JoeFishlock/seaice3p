import numpy as np
from celestine.params import Config


def calculate_temperature(enthalpy, salt, gas, cfg: Config, phase_masks):
    chi = cfg.physical_params.expansion_coefficient
    St = cfg.physical_params.stefan_number
    C = cfg.physical_params.concentration_ratio
    temperature = np.full_like(enthalpy, np.NaN)
    l, L, m, M, e, E, s, S = phase_masks
    temperature[l] = enthalpy[l]
    temperature[L] = enthalpy[L] / (1 - ((gas[L] - chi) / (1 - chi)))

    coeff1 = enthalpy[m] + C + St
    coeff2 = C * enthalpy[m] - St * salt[m]
    temperature[m] = (1 / 2) * (coeff1 - np.sqrt(coeff1**2 - 4 * coeff2))

    coeff3 = 1 - gas[M]
    coeff4 = (C + St) * (1 - gas[M]) + enthalpy[M] + salt[M] * chi + C * chi
    coeff5 = C * enthalpy[M] - (1 - chi) * St * salt[M] - C * St * (gas[M] - chi)
    temperature[M] = (1 / (2 * coeff3)) * (
        coeff4 - np.sqrt(coeff4**2 - 4 * coeff3 * coeff5)
    )

    temperature[e] = -1
    temperature[E] = -1

    temperature[s] = enthalpy[s] + St
    temperature[S] = enthalpy[S] / (1 - gas[S]) + St

    return temperature


def calculate_liquid_fraction(
    enthalpy, salt, gas, temperature, cfg: Config, phase_masks
):
    chi = cfg.physical_params.expansion_coefficient
    St = cfg.physical_params.stefan_number
    C = cfg.physical_params.concentration_ratio
    liquid_fraction = np.full_like(enthalpy, np.NaN)
    l, L, m, M, e, E, s, S = phase_masks

    liquid_fraction[l] = 1
    liquid_fraction[L] = 1 - ((gas[L] - chi) / (1 - chi))

    liquid_fraction[m] = 1 - (temperature[m] - enthalpy[m]) / St
    liquid_fraction[M] = (salt[M] + C) / (C - temperature[M])

    liquid_fraction[e] = (enthalpy[e] + 1) / St + 1
    liquid_fraction[E] = ((1 - gas[E]) * (1 + St) + enthalpy[E]) / (
        St * (1 - chi) - chi
    )

    liquid_fraction[s] = 0
    liquid_fraction[S] = 0

    return liquid_fraction


def calculate_gas_fraction(gas, liquid_fraction, cfg: Config, phase_masks):
    chi = cfg.physical_params.expansion_coefficient
    gas_fraction = np.full_like(gas, np.NaN)
    l, L, m, M, e, E, s, S = phase_masks

    gas_fraction[l] = 0
    gas_fraction[L] = (gas[L] - chi) / (1 - chi)

    gas_fraction[m] = 0
    gas_fraction[M] = gas[M] - chi * liquid_fraction[M]

    gas_fraction[e] = 0
    gas_fraction[E] = gas[E] - chi * liquid_fraction[E]

    gas_fraction[s] = 0
    gas_fraction[S] = gas[S]

    return gas_fraction


def calculate_solid_fraction(liquid_fraction, gas_fraction, cfg: Config):
    solid_fraction = 1 - liquid_fraction - gas_fraction
    return solid_fraction


def calculate_dissolved_gas(gas, liquid_fraction, cfg: Config, phase_masks):
    chi = cfg.physical_params.expansion_coefficient
    dissolved_gas = np.full_like(gas, np.NaN)
    l, L, m, M, e, E, s, S = phase_masks

    dissolved_gas[l] = gas[l] / chi
    dissolved_gas[L] = 1

    dissolved_gas[m] = gas[m] / (chi * liquid_fraction[m])
    dissolved_gas[M] = 1

    dissolved_gas[e] = gas[e] / (chi * liquid_fraction[e])
    dissolved_gas[E] = 1

    dissolved_gas[s] = 1
    dissolved_gas[S] = 1

    return dissolved_gas


def calculate_liquid_salinity(salt, gas, temperature, cfg: Config, phase_masks):
    chi = cfg.physical_params.expansion_coefficient
    C = cfg.physical_params.concentration_ratio
    liquid_salinity = np.full_like(salt, np.NaN)
    l, L, m, M, e, E, s, S = phase_masks

    liquid_salinity[l] = salt[l]
    gas_fraction = (gas[L] - chi) / (1 - chi)
    liquid_salinity[L] = (salt[L] + gas_fraction * C) / (1 - gas_fraction)

    liquid_salinity[m] = -temperature[m]
    liquid_salinity[M] = -temperature[M]

    liquid_salinity[e] = 1
    liquid_salinity[E] = 1

    liquid_salinity[s] = 1
    liquid_salinity[S] = 1

    return liquid_salinity


def calculate_enthalpy_method(enthalpy, salt, gas, cfg, phase_masks):
    temperature = calculate_temperature(enthalpy, salt, gas, cfg, phase_masks)
    liquid_fraction = calculate_liquid_fraction(
        enthalpy, salt, gas, temperature, cfg, phase_masks
    )
    gas_fraction = calculate_gas_fraction(gas, liquid_fraction, cfg, phase_masks)
    solid_fraction = calculate_solid_fraction(liquid_fraction, gas_fraction, cfg)
    liquid_salinity = calculate_liquid_salinity(
        salt, gas, temperature, cfg, phase_masks
    )
    dissolved_gas = calculate_dissolved_gas(gas, liquid_fraction, cfg, phase_masks)
    return (
        temperature,
        liquid_fraction,
        gas_fraction,
        solid_fraction,
        liquid_salinity,
        dissolved_gas,
    )
