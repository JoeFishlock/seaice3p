import numpy as np
from phase_boundaries import (
    calculate_saturation,
    get_phase_masks,
    calculate_eutectic,
    calculate_solidus,
)
from forcing import get_temperature_forcing


def calculate_enthalpy_from_temp(salt: float, gas: float, temperature: float, params):
    """Use to calculate top boundary condition from imposed temperature"""
    C = params.concentration_ratio
    St = params.stefan_number
    chi = params.expansion_coefficient
    epsilon = 1e-3

    liquid_temp_sub = -salt
    liquid_gas_fraction = (gas - chi) / (1 - chi)
    liquid_temp_super = -(salt + liquid_gas_fraction * C) / (1 - liquid_gas_fraction)
    eutectic_temp = -1 + epsilon
    solid_temp = -1 - epsilon

    eutectic_enthalpy = calculate_eutectic(np.array([salt]), np.array([0]), params)
    eutectic_enthalpy = eutectic_enthalpy[0]
    solidus_enthalpy = calculate_solidus(np.array([salt]), np.array([0]), params)
    solidus_enthalpy = solidus_enthalpy[0]

    liquid_saturation = chi
    mush_saturation = chi * (salt + C) / (C - temperature)
    alpha = epsilon / (eutectic_enthalpy - solidus_enthalpy)
    eutectic_saturation = (chi / St) * (
        (temperature + 1) / (2 * alpha)
        + 0.5 * (eutectic_enthalpy + solidus_enthalpy)
        + St
        - temperature
    )
    solid_saturation = 0

    is_liquid_sub = (temperature > liquid_temp_sub) & (gas <= liquid_saturation)
    is_liquid_super = (temperature > liquid_temp_super) & (gas > liquid_saturation)
    is_mush_sub = (
        (temperature > eutectic_temp)
        & (temperature <= liquid_temp_sub)
        & (gas <= mush_saturation)
    )
    is_mush_super = (
        (temperature > eutectic_temp)
        & (temperature <= liquid_temp_super)
        & (gas > mush_saturation)
    )
    is_eutectic_sub = (
        (temperature > solid_temp)
        & (temperature <= eutectic_temp)
        & (gas <= eutectic_saturation)
    )
    is_eutectic_super = (
        (temperature > solid_temp)
        & (temperature <= eutectic_temp)
        & (gas > eutectic_saturation)
    )
    is_solid_sub = (temperature <= solid_temp) & (gas <= solid_saturation)
    is_solid_super = (temperature <= solid_temp) & (gas > solid_saturation)

    if is_liquid_sub:
        return temperature
    if is_liquid_super:
        return (1 - liquid_gas_fraction) * temperature

    if is_mush_sub:
        return temperature - (1 - (salt + C) / (C - temperature)) * St

    if is_mush_super:
        return (1 - gas) * (temperature - St) + (salt + C) * (
            temperature * chi + 1 - chi
        ) * St / (C - temperature)

    if is_eutectic_sub or is_eutectic_super:
        return (temperature + 1) / (2 * alpha) + 0.5 * (
            eutectic_enthalpy + solidus_enthalpy
        )

    if is_solid_sub:
        return temperature - St

    if is_solid_super:
        return (1 - gas) * (temperature - St)

    raise ValueError("Cannot identify phase in top ghost cell")


def add_enthalpy_bcs(
    enthalpy_ghost, salt_ghost, gas_ghost, top_temp, bottom_temp, params
):
    """Adds enthalpy corresponding to top and bottom temperature forcing"""
    C = params.concentration_ratio
    enthalpy_ghost[-1] = calculate_enthalpy_from_temp(
        salt_ghost[-1], gas_ghost[-1], top_temp, params
    )
    enthalpy_ghost[0] = calculate_enthalpy_from_temp(
        salt_ghost[0], gas_ghost[0], bottom_temp, params
    )


def add_salt_bcs(salt_ghost, params):
    """pads centered bulk salinity with initial bulk salinity at top and bottom"""
    C = params.concentration_ratio
    salt_ghost[0] = 0
    salt_ghost[-1] = 0


def add_pressure_bcs(pressure_ghost):
    """pads centered bulk salinity with initial bulk salinity at top and bottom"""
    pressure_ghost[0] = 0
    pressure_ghost[-1] = pressure_ghost[-2]


def add_gas_bcs(gas_ghost, top_temp, bottom_temp, params):
    """Adds saturation bulk gas content at either end of the domain"""
    # C = params.concentration_ratio
    # a = params.far_gas_sat

    # top_enthalpy = calculate_enthalpy_from_temp(C, top_temp, params)
    # bottom_enthalpy = calculate_enthalpy_from_temp(C, bottom_temp, params)
    # top_enthalpy = np.array([top_enthalpy])
    # bottom_enthalpy = np.array([bottom_enthalpy])

    # top_phase_masks = get_phase_masks(top_enthalpy, np.array([C]), params)
    # top_saturation_concentration = calculate_saturation_concentration(
    #     top_enthalpy, np.array([C]), top_phase_masks, params
    # )
    # top_liquid_fraction = calculate_liquid_fraction(
    #     top_enthalpy, np.array([C]), np.array([top_temp]), top_phase_masks, params
    # )
    # top_gas = calculate_saturation_boundary(
    #     top_liquid_fraction, top_saturation_concentration, params
    # )

    # bottom_phase_masks = get_phase_masks(bottom_enthalpy, np.array([C]), params)
    # bottom_saturation_concentration = calculate_saturation_concentration(
    #     bottom_enthalpy, np.array([C]), bottom_phase_masks, params
    # )
    # bottom_liquid_fraction = calculate_liquid_fraction(
    #     bottom_enthalpy,
    #     np.array([C]),
    #     np.array([bottom_temp]),
    #     bottom_phase_masks,
    #     params,
    # )
    # bottom_gas = a * calculate_saturation_boundary(
    #     bottom_liquid_fraction, bottom_saturation_concentration, params
    # )
    # TODO: Need to implement this fully but shouldn't matter when gas diffusion is turned off <13-03-23, Joe Fishlock> #
    bottom_gas = np.array([params.expansion_coefficient])
    top_gas = np.array([0])
    gas_ghost[0] = bottom_gas[0]
    gas_ghost[-1] = top_gas[0]


def apply_boundary_conditions(enthalpy, salt, gas, pressure, time, params):
    """Boundary conditions applied to arrays already on ghost grid"""
    top_temperature = get_temperature_forcing(time, params)
    add_enthalpy_bcs(enthalpy, salt, gas, top_temperature, params.far_temp, params)
    add_salt_bcs(salt, params)
    add_gas_bcs(gas, top_temperature, params.far_temp, params)
    add_pressure_bcs(pressure)
