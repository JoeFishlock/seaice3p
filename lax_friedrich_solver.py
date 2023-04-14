import numpy as np
from boundary_conditions import calculate_enthalpy_from_temp
from forcing import get_temperature_forcing
from enthalpy_method import (
    calculate_enthalpy_method,
    get_phase_masks,
)
from grids import (
    get_difference_matrix,
    upwind,
    geometric,
    initialise_grids,
    average,
)
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


def take_timestep(enthalpy, salt, gas, pressure, time, timestep, params, D_e, D_g):
    I = params.I
    chi = params.expansion_coefficient
    C = params.concentration_ratio

    new_time = time + timestep
    new_top_temperature = get_temperature_forcing(new_time, params)
    new_top_enthalpy = calculate_enthalpy_from_temp(0, 0, new_top_temperature, params)

    phase_masks = get_phase_masks(
        enthalpy,
        salt,
        gas,
        params,
    )
    (
        temperature,
        liquid_fraction,
        gas_fraction,
        _,
        liquid_salinity,
        dissolved_gas,
    ) = calculate_enthalpy_method(enthalpy, salt, gas, params, phase_masks)
    Vg, Wl, V = calculate_velocities(
        liquid_fraction, enthalpy, salt, gas, pressure, D_g, params
    )

    new_enthalpy = np.zeros((I + 2,))
    new_salt = np.zeros((I + 2,))
    new_gas = np.zeros((I + 2,))
    new_pressure = np.zeros((I + 2,))

    new_enthalpy[-1] = new_top_enthalpy
    new_enthalpy[0] = params.far_temp
    new_salt[-1] = 0
    new_salt[0] = 0
    new_gas[-1] = 0
    new_gas[0] = chi

    # Upwinding
    upwind_enthalpy_flux = (
        -(D_g @ temperature) + upwind(temperature, Wl) + upwind(enthalpy, V)
    )
    upwind_salt_flux = (
        -(1 / params.lewis_salt)
        * (geometric(liquid_fraction) * (D_g @ liquid_salinity))
        + upwind(salt, V)
        + upwind(liquid_salinity + C, Wl)
    )
    upwind_gas_flux = (
        -(chi / params.lewis_gas) * (geometric(liquid_fraction) * (D_g @ dissolved_gas))
        + upwind(gas, V)
        + upwind(gas_fraction, Vg)
        + upwind(chi * dissolved_gas, Wl)
    )

    # Lax Friedrich
    salt_no_flux = np.insert(salt[1:-1], 0, salt[0])
    salt_no_flux = np.append(salt_no_flux, salt[-1])

    gas_no_flux = np.insert(gas[1:-1], 0, gas[0])
    gas_no_flux = np.append(gas_no_flux, gas[-1])

    numerical_diffusivity = (params.step**2) / (2 * timestep)

    salt_frame_advection = V * average(salt)
    salt_liquid_advection = Wl * (average(liquid_salinity) + C)
    salt_diffusion = (
        -(1 / params.lewis_salt) * geometric(liquid_fraction) * (D_g @ liquid_salinity)
    )
    salt_LF_diffusion = -numerical_diffusivity * (D_g @ salt_no_flux)
    LF_salt_flux = (
        salt_frame_advection
        + salt_liquid_advection
        + salt_diffusion
        + salt_LF_diffusion
    )

    gas_frame_advection = V * average(gas)
    gas_bubble_advection = average(gas_fraction) * Vg
    gas_liquid_advection = chi * Wl * average(dissolved_gas)
    gas_diffusion = (
        -(chi / params.lewis_gas) * geometric(liquid_fraction) * (D_g @ dissolved_gas)
    )
    gas_LF_diffusion = -numerical_diffusivity * (D_g @ gas_no_flux)
    LF_gas_flux = (
        gas_frame_advection
        + gas_bubble_advection
        + gas_liquid_advection
        + gas_diffusion
        + gas_LF_diffusion
    )
    # this works to prevent salt diffusion in solid and for small gas bubbles
    # However for more general case condition for gas flux should perhaps be when
    # R_B = R_T
    is_solid = geometric(liquid_fraction) == 0
    # Must always upwind on the boundaries to avoid using incomplete information here
    is_solid[-1] = True
    is_solid[0] = True

    enthalpy_flux = upwind_enthalpy_flux
    salt_flux = np.where(is_solid, upwind_salt_flux, LF_salt_flux)
    gas_flux = np.where(is_solid, upwind_gas_flux, LF_gas_flux)

    new_enthalpy[1:-1] = enthalpy[1:-1] - timestep * (D_e @ enthalpy_flux)
    new_salt[1:-1] = salt[1:-1] - timestep * (D_e @ salt_flux)
    new_gas[1:-1] = gas[1:-1] - timestep * (D_e @ gas_flux)

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
    # toggle this to old liquid fraction to see if it makes any difference
    new_permeability = calculate_absolute_permeability(geometric(new_liquid_fraction))

    pressure_forcing = np.zeros((I + 2,))
    pressure_forcing[1:-1] = (1 / timestep) * (
        new_gas_fraction[1:-1] - gas_fraction[1:-1]
    ) + np.matmul(D_e, upwind(new_gas_fraction, V))
    pressure_forcing[0] = 0
    pressure_forcing[-1] = 0
    pressure_matrix = np.zeros((I + 2, I + 2))
    perm_matrix = np.zeros((I + 1, I + 1))
    np.fill_diagonal(perm_matrix, new_permeability + 1e-15)
    pressure_matrix[1:-1, :] = np.matmul(D_e, np.matmul(-perm_matrix, D_g))
    pressure_matrix[0, 0] = 1
    pressure_matrix[-1, -1] = 1
    pressure_matrix[-1, -2] = -1
    new_pressure = np.linalg.solve(pressure_matrix, pressure_forcing)

    step, _, _, _ = initialise_grids(params.I)

    # Calculation for adaptive timestepping
    CFL_timesteps = (
        params.CFL_limit
        * step
        / np.where(np.abs(upwind(gas_fraction, Vg)) > 0, np.abs(Vg), 1e-10)
    )
    Courant_timesteps = params.Courant_limit * step**2
    CFL_min_timestep = np.min(CFL_timesteps)
    Courant_min_timestep = np.min(Courant_timesteps)
    min_timestep = min(CFL_min_timestep, Courant_min_timestep)

    return new_enthalpy, new_salt, new_gas, new_pressure, new_time, min_timestep


def advance(enthalpy, salt, gas, pressure, time, timestep, params, D_e, D_g):
    (
        new_enthalpy,
        new_salt,
        new_gas,
        new_pressure,
        new_time,
        min_timestep,
    ) = take_timestep(enthalpy, salt, gas, pressure, time, timestep, params, D_e, D_g)
    while timestep > min_timestep:
        timestep = min_timestep
        (
            new_enthalpy,
            new_salt,
            new_gas,
            new_pressure,
            new_time,
            min_timestep,
        ) = take_timestep(
            enthalpy, salt, gas, pressure, time, timestep, params, D_e, D_g
        )

    return (
        new_enthalpy,
        new_salt,
        new_gas,
        new_pressure,
        new_time,
        timestep,
        min_timestep,
    )


def solve(params):
    enthalpy, salt, gas, pressure = generate_initial_solution(params, params.I + 2)
    T = params.total_time
    timestep = params.timestep
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
    time = 0
    while time < T:
        enthalpy, salt, gas, pressure, time, timestep, min_timestep = advance(
            enthalpy, salt, gas, pressure, time, timestep, params, D_e, D_g
        )
        time_to_save += timestep
        print(
            f"time={time:.3f}/{params.total_time}, timestep={timestep:.2g} \r", end=""
        )
        if np.min(salt) < -params.concentration_ratio:
            raise ValueError("salt crash")

        timestep = min_timestep

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
    # clear line after carriage return
    print("")
    return 0, "solve complete"
