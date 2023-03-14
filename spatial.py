import numpy as np
from enthalpy_method import calculate_enthalpy_method, get_phase_masks
from grids import upwind, geometric
from velocities import calculate_velocities
from boundary_conditions import calculate_enthalpy_from_temp


def calculate_discretised_fluxes(enthalpy, salt, gas, pressure, params, D_e, D_g):
    St = params.stefan_number
    C = params.concentration_ratio
    X = params.expansion_coefficient
    """Use enthalpy method to get all quantities needed"""
    phase_masks = get_phase_masks(enthalpy, salt, gas, params)
    (
        temperature,
        liquid_fraction,
        gas_fraction,
        solid_fraction,
        liquid_salinity,
        dissolved_gas,
    ) = calculate_enthalpy_method(enthalpy, salt, gas, params, phase_masks)

    """so that Vg doesn't bring in gas fraction from top"""
    gas_fraction[-1] = 0
    if np.any(gas_fraction > 0.5):
        raise ValueError("gas fraction blow up")

    Vg, Wl, V = calculate_velocities(liquid_fraction, pressure, D_g, params)

    """calculate fluxes"""
    # TODO: check these from non dimensionalisation <13-03-23, Joe Fishlock> #
    enthalpy_fluxes = (
        np.matmul(D_e, np.matmul(D_g, temperature))
        - np.matmul(D_e, upwind(temperature, Wl))
        - np.matmul(D_e, upwind(enthalpy, V))
    )
    salt_fluxes = (
        (1 / params.lewis_salt)
        * np.matmul(D_e, geometric(liquid_fraction) * np.matmul(D_g, liquid_salinity))
        - np.matmul(D_e, upwind(salt, V))
        - np.matmul(D_e, upwind(liquid_salinity + C, Wl))
    )
    gas_fluxes = (
        (X / params.lewis_gas)
        * np.matmul(D_e, geometric(liquid_fraction) * np.matmul(D_g, dissolved_gas))
        - np.matmul(D_e, upwind(gas, V))
        - np.matmul(D_e, upwind(gas_fraction, Vg))
        - np.matmul(D_e, upwind(X * dissolved_gas, Wl))
    )

    return enthalpy_fluxes, salt_fluxes, gas_fluxes


def generate_initial_solution(params, length):
    """Generate initial solution on the ghost grid"""
    bottom_enthalpy = calculate_enthalpy_from_temp(
        params.concentration_ratio,
        params.expansion_coefficient * params.far_gas_sat,
        params.far_temp,
        params,
    )
    enthalpy = np.full((length,), bottom_enthalpy)
    salt = np.full_like(enthalpy, 0)
    gas = np.full_like(enthalpy, params.expansion_coefficient * params.far_gas_sat)
    pressure = np.full_like(enthalpy, 0)
    return enthalpy, salt, gas, pressure
