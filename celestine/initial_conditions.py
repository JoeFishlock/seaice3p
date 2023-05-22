"""Provide initial state of bulk enthalpy, bulk salinity and bulk gas for the
simulation.
"""
import numpy as np
from celestine.params import Config
from celestine.state import State


def get_initial_conditions(cfg: Config):
    choice = cfg.boundary_conditions_config.initial_conditions_choice
    return INITIAL_CONDITIONS[choice](cfg)


def get_uniform_initial_conditions(cfg):
    """Generate uniform initial solution on the ghost grid

    :returns: initial solution arrays on ghost grid (enthalpy, salt, gas, pressure)
    """
    chi = cfg.physical_params.expansion_coefficient

    bottom_temp = cfg.boundary_conditions_config.far_temp
    bottom_bulk_salinity = cfg.boundary_conditions_config.far_bulk_salinity
    bottom_dissolved_gas = cfg.boundary_conditions_config.far_gas_sat
    bottom_bulk_gas = bottom_dissolved_gas * chi

    # Initialise uniform enthalpy assuming completely liquid initial domain
    enthalpy = np.full((cfg.numerical_params.I,), bottom_temp)
    salt = np.full_like(enthalpy, bottom_bulk_salinity)
    gas = np.full_like(enthalpy, bottom_bulk_gas)
    pressure = np.full_like(enthalpy, 0)

    initial_state = State(cfg, 0, enthalpy, salt, gas, pressure)

    return initial_state


INITIAL_CONDITIONS = {
    "uniform": get_uniform_initial_conditions,
}
