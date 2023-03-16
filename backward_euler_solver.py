from scipy.optimize import root
import numpy as np
from tqdm import tqdm
from boundary_conditions import calculate_enthalpy_from_temp
from forcing import get_temperature_forcing
from enthalpy_method import (
    calculate_enthalpy_method,
    get_phase_masks,
    calculate_temperature,
    calculate_liquid_fraction,
    calculate_gas_fraction,
)
from grids import get_difference_matrix, get_number_of_timesteps, upwind, geometric
from velocities import calculate_velocities, calculate_absolute_permeability


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
    I = params.I
    chi = params.expansion_coefficient
    C = params.concentration_ratio

    new_time = time + timestep
    top_temperature = get_temperature_forcing(time, params)
    new_top_temperature = get_temperature_forcing(new_time, params)
    top_enthalpy = calculate_enthalpy_from_temp(0, 0, top_temperature, params)
    new_top_enthalpy = calculate_enthalpy_from_temp(0, 0, new_top_temperature, params)

    """Extend to ghost grids"""
    solution_ghost = np.zeros((4 * (I + 2)))
    new_solution_ghost = np.zeros((4 * (I + 2)))

    solution_ghost[1 : I + 1] = solution[0:I]
    solution_ghost[I + 3 : 2 * I + 3] = solution[I : 2 * I]
    solution_ghost[2 * I + 5 : 3 * I + 5] = solution[2 * I : 3 * I]
    solution_ghost[3 * I + 7 : 4 * I + 7] = solution[3 * I :]

    new_solution_ghost[1 : I + 1] = new_solution[0:I]
    new_solution_ghost[I + 3 : 2 * I + 3] = new_solution[I : 2 * I]
    new_solution_ghost[2 * I + 5 : 3 * I + 5] = new_solution[2 * I : 3 * I]
    new_solution_ghost[3 * I + 7 : 4 * I + 7] = new_solution[3 * I :]

    """Apply boundary conditions"""
    solution_ghost[0] = params.far_temp
    solution_ghost[I + 1] = top_enthalpy
    solution_ghost[I + 2] = 0
    solution_ghost[2 * I + 3] = 0
    solution_ghost[2 * I + 4] = params.expansion_coefficient
    solution_ghost[3 * I + 5] = 0
    solution_ghost[3 * I + 6] = 0
    solution_ghost[-1] = solution_ghost[-2]

    new_solution_ghost[0] = params.far_temp
    new_solution_ghost[I + 1] = new_top_enthalpy
    new_solution_ghost[I + 2] = 0
    new_solution_ghost[2 * I + 3] = 0
    new_solution_ghost[2 * I + 4] = params.expansion_coefficient
    new_solution_ghost[3 * I + 5] = 0
    new_solution_ghost[3 * I + 6] = 0
    new_solution_ghost[-1] = new_solution_ghost[-2]

    """calculate gas fraction for time derivative in pressure residual"""
    enthalpy = solution_ghost[0 : I + 2]
    salt = solution_ghost[I + 2 : 2 * I + 4]
    gas = solution_ghost[2 * I + 4 : 3 * I + 6]
    phase_masks = get_phase_masks(
        enthalpy,
        salt,
        gas,
        params,
    )
    temperature = calculate_temperature(enthalpy, salt, gas, params, phase_masks)
    liquid_fraction = calculate_liquid_fraction(
        enthalpy, salt, gas, temperature, params, phase_masks
    )
    gas_fraction = calculate_gas_fraction(gas, liquid_fraction, params, phase_masks)

    """calculate variables needed to compute the fluxes at new timestep"""
    new_enthalpy = new_solution_ghost[0 : I + 2]
    new_salt = new_solution_ghost[I + 2 : 2 * I + 4]
    new_gas = new_solution_ghost[2 * I + 4 : 3 * I + 6]
    new_pressure = new_solution_ghost[3 * I + 6 :]
    new_phase_masks = get_phase_masks(
        new_enthalpy,
        new_salt,
        new_gas,
        params,
    )
    (
        new_temperature,
        new_liquid_fraction,
        new_gas_fraction,
        _,
        new_liquid_salinity,
        new_dissolved_gas,
    ) = calculate_enthalpy_method(
        new_enthalpy, new_salt, new_gas, params, new_phase_masks
    )
    Vg, Wl, V = calculate_velocities(new_liquid_fraction, new_pressure, D_g, params)
    new_permeability = calculate_absolute_permeability(geometric(new_liquid_fraction))

    """initialise time derivatives enthalpy, salt, gas, gas_fraction on centers"""
    time_derivs = np.zeros((4 * I,))
    time_derivs[0 : 3 * I] = (1 / timestep) * (
        new_solution[0 : 3 * I] - solution[0 : 3 * I]
    )
    time_derivs[3 * I :] = (1 / timestep) * (
        new_gas_fraction[1:-1] - gas_fraction[1:-1]
    )

    """initialise ode_functions"""
    ode_func = np.zeros((4 * I,))
    ode_func[0:I] = (
        np.matmul(D_e, np.matmul(D_g, new_temperature))
        - np.matmul(D_e, upwind(new_temperature, Wl))
        - np.matmul(D_e, upwind(new_enthalpy, V))
    )
    ode_func[I : 2 * I] = (
        (1 / params.lewis_salt)
        * np.matmul(
            D_e, geometric(new_liquid_fraction) * np.matmul(D_g, new_liquid_salinity)
        )
        - np.matmul(D_e, upwind(new_salt, V))
        - np.matmul(D_e, upwind(new_liquid_salinity + C, Wl))
    )
    ode_func[2 * I : 3 * I] = (
        (chi / params.lewis_gas)
        * np.matmul(
            D_e, geometric(new_liquid_fraction) * np.matmul(D_g, new_dissolved_gas)
        )
        - np.matmul(D_e, upwind(new_gas, V))
        - np.matmul(D_e, upwind(new_gas_fraction, Vg))
        - np.matmul(D_e, upwind(chi * new_dissolved_gas, Wl))
    )
    ode_func[3 * I :] = -np.matmul(D_e, upwind(new_gas_fraction, V)) + np.matmul(
        D_e, (-new_permeability - 1e-7) * np.matmul(D_g, new_pressure)
    )

    return time_derivs - ode_func


def solve_non_linear_system(solution, time, params, D_e, D_g):
    new_solution = root(
        lambda x: calculate_residual(x, solution, time, params, D_e, D_g),
        x0=solution,
        method="krylov",
        options={"maxiter": 5},
    )
    return new_solution.x


def solve(params):
    if params.check_buoyancy_CFL():
        return 1, "unstable CFL for gas advection"

    if params.check_thermal_diffusion_stability():
        return 1, "unstable thermal diffusion"

    enthalpy, salt, gas, pressure = generate_initial_solution(params, params.I + 2)
    N = get_number_of_timesteps(params.total_time, params.timestep)
    D_e = get_difference_matrix(params.I, params.step)
    D_g = get_difference_matrix(params.I + 1, params.step)

    """generate initial solutions on centers"""
    enthalpy = enthalpy[1:-1]
    salt = salt[1:-1]
    gas = gas[1:-1]
    pressure = pressure[1:-1]
    (
        stored_times,
        stored_enthalpy,
        stored_salt,
        stored_gas,
        stored_pressure,
    ) = generate_storage_arrays(enthalpy, salt, gas, pressure)
    solution = np.hstack((enthalpy, salt, gas, pressure))
    time_to_save = 0
    for n in tqdm(range(N)):
        time = n * params.timestep
        time_to_save += params.timestep
        solution = solve_non_linear_system(solution, time, params, D_e, D_g)

        if (time_to_save - params.savefreq) >= 0:
            enthalpy, salt, gas, pressure = np.split(solution, 4)
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
