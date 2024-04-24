"""Module to provide initial state of bulk enthalpy, bulk salinity and bulk gas for the
simulation.
"""
import numpy as np
from celestine.params import Config
from celestine.state import EQMState
from celestine.grids import initialise_grids


def get_initial_conditions(cfg: Config):
    INITIAL_CONDITIONS = {
        "uniform": get_uniform_initial_conditions,
        "barrow_2009": get_barrow_initial_conditions,
    }
    choice = cfg.boundary_conditions_config.initial_conditions_choice
    return INITIAL_CONDITIONS[choice](cfg)


def get_uniform_initial_conditions(cfg):
    """Generate uniform initial solution on the ghost grid

    :returns: initial solution arrays on ghost grid (enthalpy, salt, gas)
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

    initial_state = EQMState(cfg, 0, enthalpy, salt, gas)

    return initial_state


def apply_value_in_ice_layer(depth_of_ice, ice_value, liquid_value, grid):
    """assume that top part of domain contains mushy ice of given depth and lower part
    of domain is liquid. This function returns output on the given grid where the ice
    part of the domain takes one value and the liquid a different.

    This is useful for initialising the barrow simulation where we have an initial ice
    layer.
    """
    output = np.where(grid > -depth_of_ice, ice_value, liquid_value)
    return output


def get_barrow_initial_conditions(cfg: Config):
    """initialise domain with an initial ice layer of given temperature and bulk
    salinity. These values are hard coded in from Moreau paper to match barrow study.
    They also assume that the initial ice layer has 1/5 the saturation amount in pure
    liquid of dissolved gas to account for previous gas loss.

    Initialise with bulk gas being (1/5) in ice and saturation in liquid.
    Bulk salinity is 5.92 g/kg in ice and ocean value in liquid.
    Enthalpy is calculated by inverting temperature relation in ice and ocean.
    Ice temperature is given as -8.15 degC and ocean is the far value from boundary
    config.
    """
    far_gas_sat = cfg.boundary_conditions_config.far_gas_sat
    ICE_DEPTH = cfg.scales.convert_from_dimensional_grid(0.7)

    # if we are going to have brine convection ice will desalinate on its own
    if cfg.darcy_law_params.brine_convection_parameterisation:
        SALT_IN_ICE = cfg.boundary_conditions_config.far_bulk_salinity
    else:
        SALT_IN_ICE = cfg.scales.convert_from_dimensional_bulk_salinity(5.92)

    BOTTOM_TEMP = cfg.scales.convert_from_dimensional_temperature(-1.8)
    BOTTOM_SALT = cfg.boundary_conditions_config.far_bulk_salinity
    TEMP_IN_ICE = cfg.scales.convert_from_dimensional_temperature(-8.15)

    chi = cfg.physical_params.expansion_coefficient

    _, centers, _, _ = initialise_grids(cfg.numerical_params.I)
    salt = apply_value_in_ice_layer(
        ICE_DEPTH, ice_value=SALT_IN_ICE, liquid_value=BOTTOM_SALT, grid=centers
    )
    gas = apply_value_in_ice_layer(
        ICE_DEPTH,
        ice_value=cfg.forcing_config.Barrow_initial_bulk_gas_in_ice * chi,
        liquid_value=chi * far_gas_sat,
        grid=centers,
    )

    temp = apply_value_in_ice_layer(
        ICE_DEPTH, ice_value=TEMP_IN_ICE, liquid_value=BOTTOM_TEMP, grid=centers
    )
    solid_fraction_in_mush = (salt + temp) / (
        temp - cfg.physical_params.concentration_ratio
    )
    enthalpy = apply_value_in_ice_layer(
        ICE_DEPTH,
        ice_value=temp - solid_fraction_in_mush * cfg.physical_params.stefan_number,
        liquid_value=temp,
        grid=centers,
    )

    initial_state = EQMState(cfg, 0, enthalpy, salt, gas)

    return initial_state
