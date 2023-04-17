import numpy as np
from celestine.boundary_conditions import calculate_enthalpy_from_temp
from celestine.params import Config


def generate_initial_solution(cfg: Config, length):
    """Generate initial solution on the ghost grid"""
    bottom_enthalpy = calculate_enthalpy_from_temp(
        cfg.physical_params.concentration_ratio,
        cfg.physical_params.expansion_coefficient
        * cfg.boundary_conditions_config.far_gas_sat,
        cfg.boundary_conditions_config.far_temp,
        cfg,
    )
    enthalpy = np.full((length,), bottom_enthalpy)
    salt = np.full_like(enthalpy, 0)
    gas = np.full_like(
        enthalpy,
        cfg.physical_params.expansion_coefficient
        * cfg.boundary_conditions_config.far_gas_sat,
    )
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
    stored_times, stored_enthalpy, stored_salt, stored_gas, stored_pressure, cfg: Config
):
    np.savez(
        f"{cfg.data_path}{cfg.name}.npz",
        times=stored_times,
        enthalpy=np.transpose(stored_enthalpy),
        salt=np.transpose(stored_salt),
        gas=np.transpose(stored_gas),
        pressure=np.transpose(stored_pressure),
    )
