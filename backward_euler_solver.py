from scipy.optimize import root
import numpy as np
from tqdm import tqdm
from boundary_conditions import (
    apply_boundary_conditions,
)
from enthalpy_method import calculate_enthalpy_method, get_phase_masks
from grids import get_difference_matrix, get_number_of_timesteps, upwind, geometric
from spatial import calculate_discretised_fluxes, generate_initial_solution
from velocities import calculate_velocities, calculate_absolute_permeability


def generate_storage_arrays(enthalpy, salt, gas, pressure):
    stored_enthalpy = np.copy(enthalpy)
    stored_salt = np.copy(salt)
    stored_gas = np.copy(gas)
    stored_pressure = np.copy(pressure)
    stored_times = np.array([0])
    return stored_times, stored_enthalpy, stored_salt, stored_gas, stored_pressure


def save_storage(
    stored_times, stored_enthalpy, stored_salt, stored_gas, stored_pressure, params
):
    np.savez(
        f"{params.data_path}{params.name}.npz",
        times=stored_times,
        enthalpy=np.transpose(stored_enthalpy),
        salt=np.transpose(stored_salt),
        gas=np.transpose(stored_gas),
        pressure=np.transpose(stored_pressure),
    )


def calculate_residual(new_solution, solution, time, params, D_e, D_g):
    timestep = params.timestep

    new_enthalpy, new_salt, new_gas, new_pressure = np.split(new_solution, 4)
    new_enthalpy = np.concatenate((np.array([0]), new_enthalpy, np.array([0])))
    new_salt = np.concatenate((np.array([0]), new_salt, np.array([0])))
    new_gas = np.concatenate((np.array([0]), new_gas, np.array([0])))
    new_pressure = np.concatenate((np.array([0]), new_pressure, np.array([0])))
    apply_boundary_conditions(
        new_enthalpy, new_salt, new_gas, new_pressure, time, params
    )
    enthalpy, salt, gas, pressure = np.split(solution, 4)

    enthalpy_flux, salt_flux, gas_flux = calculate_discretised_fluxes(
        new_enthalpy, new_salt, new_gas, new_pressure, params, D_e, D_g
    )

    enthalpy_residual = (new_enthalpy[1:-1] - enthalpy[1:-1]) / timestep - enthalpy_flux
    salt_residual = (new_salt[1:-1] - salt[1:-1]) / timestep - salt_flux
    gas_residual = (new_gas[1:-1] - gas[1:-1]) / timestep - gas_flux

    new_phase_masks = get_phase_masks(new_enthalpy, new_salt, new_gas, params)
    (
        new_temperature,
        new_liquid_fraction,
        new_gas_fraction,
        new_solid_fraction,
        new_liquid_salinity,
        new_dissolved_gas,
    ) = calculate_enthalpy_method(
        new_enthalpy, new_salt, new_gas, params, new_phase_masks
    )
    Vg, Wl, V = calculate_velocities(new_liquid_fraction, new_pressure, D_g, params)
    new_permeability = calculate_absolute_permeability(geometric(new_liquid_fraction))
    phase_masks = get_phase_masks(enthalpy, salt, gas, params)
    (
        temperature,
        liquid_fraction,
        gas_fraction,
        solid_fraction,
        liquid_salinity,
        dissolved_gas,
    ) = calculate_enthalpy_method(enthalpy, salt, gas, params, phase_masks)
    pressure_residual = (
        (new_gas_fraction[1:-1] - gas_fraction[1:-1]) / timestep
        + np.matmul(D_e, upwind(new_gas_fraction, V))
        - np.matmul(D_e, (-new_permeability - 1e-7) * np.matmul(D_g, new_pressure))
    )
    return np.hstack(
        (enthalpy_residual, salt_residual, gas_residual, pressure_residual)
    )


def solve_non_linear_system(solution, time, params, D_e, D_g):
    enthalpy, salt, gas, pressure = np.split(solution, 4)
    guess = np.hstack((enthalpy[1:-1], salt[1:-1], gas[1:-1], pressure[1:-1]))
    new_solution = root(
        lambda x: calculate_residual(x, solution, time, params, D_e, D_g),
        x0=guess,
        method="krylov",
        options={"maxiter": 3},
    )
    return new_solution


def take_timestep(enthalpy, salt, gas, pressure, time, params, D_e, D_g):
    timestep = params.timestep

    apply_boundary_conditions(enthalpy, salt, gas, pressure, time, params)

    """initialise new solution vectors on ghost grid"""
    enthalpy_new = np.copy(enthalpy)
    salt_new = np.copy(salt)
    gas_new = np.copy(gas)
    pressure_new = np.copy(pressure)

    solution = np.hstack((enthalpy_new, salt_new, gas_new, pressure_new))
    solution_new = solve_non_linear_system(solution, time, params, D_e, D_g)
    enthalpy_new[1:-1], salt_new[1:-1], gas_new[1:-1], pressure_new[1:-1] = np.split(
        solution_new.x, 4
    )
    return enthalpy_new, salt_new, gas_new, pressure_new


def solve(params):
    if params.check_buoyancy_CFL():
        return 1, "unstable CFL for gas advection"

    if params.check_thermal_diffusion_stability():
        return 1, "unstable thermal diffusion"

    enthalpy, salt, gas, pressure = generate_initial_solution(params, params.I + 2)
    N = get_number_of_timesteps(params.total_time, params.timestep)
    D_e = get_difference_matrix(params.I, params.step)
    D_g = get_difference_matrix(params.I + 1, params.step)

    (
        stored_times,
        stored_enthalpy,
        stored_salt,
        stored_gas,
        stored_pressure,
    ) = generate_storage_arrays(enthalpy, salt, gas, pressure)
    time_to_save = 0
    for n in tqdm(range(N)):
        time = n * params.timestep
        time_to_save += params.timestep
        enthalpy, salt, gas, pressure = take_timestep(
            enthalpy, salt, gas, pressure, time, params, D_e, D_g
        )

        if (time_to_save - params.savefreq) >= 0:
            time_to_save = 0
            stored_times = np.append(stored_times, time)
            stored_enthalpy = np.vstack((stored_enthalpy, enthalpy))
            stored_salt = np.vstack((stored_salt, salt))
            stored_gas = np.vstack((stored_gas, gas))
            stored_pressure = np.vstack((stored_pressure, pressure))

    save_storage(
        stored_times, stored_enthalpy, stored_salt, stored_gas, stored_pressure, params
    )
    return 0, "solve complete"
