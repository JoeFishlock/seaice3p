"""Module to provide initial state of bulk enthalpy, bulk salinity and bulk gas for the
simulation.
"""
import numpy as np

from .params import (
    Config,
    UniformInitialConditions,
    BRW09InitialConditions,
    OilInitialConditions,
    NoBrineConvection,
    EQMPhysicalParams,
    DISEQPhysicalParams,
    PreviousSimulation,
)
from .state import EQMState, DISEQState, State
from .grids import Grids


def get_initial_conditions(cfg: Config):
    INITIAL_CONDITIONS = {
        UniformInitialConditions: _get_uniform_initial_conditions,
        BRW09InitialConditions: _get_barrow_initial_conditions,
        OilInitialConditions: _get_oil_initial_conditions,
        PreviousSimulation: _get_previous_simulation_final_state,
    }
    initial_state = INITIAL_CONDITIONS[type(cfg.initial_conditions_config)](cfg)
    match cfg.physical_params:
        case EQMPhysicalParams():
            return np.hstack(
                (initial_state.enthalpy, initial_state.salt, initial_state.gas)
            )
        case DISEQPhysicalParams():
            return np.hstack(
                (
                    initial_state.enthalpy,
                    initial_state.salt,
                    initial_state.bulk_dissolved_gas,
                    initial_state.gas_fraction,
                )
            )
        case _:
            raise NotImplementedError


def _apply_value_in_ice_layer(depth_of_ice, ice_value, liquid_value, grid):
    """assume that top part of domain contains mushy ice of given depth and lower part
    of domain is liquid. This function returns output on the given grid where the ice
    part of the domain takes one value and the liquid a different.

    This is useful for initialising the barrow simulation where we have an initial ice
    layer.
    """
    output = np.where(grid > -depth_of_ice, ice_value, liquid_value)
    return output


def _get_previous_simulation_final_state(cfg: Config):
    """Generate initial state from the final state of a saved simulation

    :returns: initial solution arrays on ghost grid (enthalpy, salt, gas)
    """
    with np.load(cfg.initial_conditions_config.data_path) as data:
        match cfg.physical_params:
            case EQMPhysicalParams():
                enthalpy = data["arr_1"]
                salt = data["arr_2"]
                bulk_gas = data["arr_3"]

                return EQMState(0, enthalpy[:, -1], salt[:, -1], bulk_gas[:, -1])

            case DISEQPhysicalParams():
                enthalpy = data["arr_1"]
                salt = data["arr_2"]
                bulk_dissolved_gas = data["arr_3"]
                gas_fraction = data["arr_4"]

                return DISEQState(
                    0,
                    enthalpy[:, -1],
                    salt[:, -1],
                    bulk_dissolved_gas[:, -1],
                    gas_fraction[:, -1],
                )

            case _:
                raise NotImplementedError


def _get_uniform_initial_conditions(cfg: Config):
    """Generate uniform initial solution on the ghost grid

    :returns: initial solution arrays on ghost grid (enthalpy, salt, gas)
    """
    chi = cfg.physical_params.expansion_coefficient

    bottom_temp = cfg.ocean_forcing_config.ocean_temp
    bottom_bulk_salinity = 0
    bottom_dissolved_gas = cfg.ocean_forcing_config.ocean_gas_sat
    bottom_bulk_gas = bottom_dissolved_gas * chi

    # Initialise uniform enthalpy assuming completely liquid initial domain
    enthalpy = np.full((cfg.numerical_params.I,), bottom_temp)
    salt = np.full_like(enthalpy, bottom_bulk_salinity)
    gas = np.full_like(enthalpy, bottom_bulk_gas)

    return _pack_initial_state(cfg, enthalpy, salt, gas)


def _get_barrow_initial_conditions(cfg: Config):
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
    far_gas_sat = cfg.ocean_forcing_config.ocean_gas_sat
    ICE_DEPTH = cfg.scales.convert_from_dimensional_grid(0.7)

    # if we are going to have brine convection ice will desalinate on its own
    if not isinstance(cfg.brine_convection_params, NoBrineConvection):
        SALT_IN_ICE = 0
    else:
        SALT_IN_ICE = cfg.scales.convert_from_dimensional_bulk_salinity(5.92)

    BOTTOM_TEMP = cfg.scales.convert_from_dimensional_temperature(-1.8)
    BOTTOM_SALT = 0
    TEMP_IN_ICE = cfg.scales.convert_from_dimensional_temperature(-8.15)

    chi = cfg.physical_params.expansion_coefficient

    centers = Grids(cfg.numerical_params.I).centers
    salt = _apply_value_in_ice_layer(
        ICE_DEPTH, ice_value=SALT_IN_ICE, liquid_value=BOTTOM_SALT, grid=centers
    )
    gas = _apply_value_in_ice_layer(
        ICE_DEPTH,
        ice_value=cfg.initial_conditions_config.Barrow_initial_bulk_gas_in_ice * chi,
        liquid_value=chi * far_gas_sat,
        grid=centers,
    )

    temp = _apply_value_in_ice_layer(
        ICE_DEPTH, ice_value=TEMP_IN_ICE, liquid_value=BOTTOM_TEMP, grid=centers
    )
    solid_fraction_in_mush = (salt + temp) / (
        temp - cfg.physical_params.concentration_ratio
    )
    enthalpy = _apply_value_in_ice_layer(
        ICE_DEPTH,
        ice_value=temp - solid_fraction_in_mush * cfg.physical_params.stefan_number,
        liquid_value=temp,
        grid=centers,
    )

    return _pack_initial_state(cfg, enthalpy, salt, gas)


def _get_oil_initial_conditions(cfg: Config):
    """initialise domain with an initial ice layer of given temperature and bulk
    salinity given by values in the configuration.

    This is an idealised initial condition to investigate the impact of shortwave
    radiative forcing on melting bare ice
    """
    ICE_DEPTH = cfg.initial_conditions_config.initial_ice_depth

    # Initialise with a constant bulk salinity in ice
    SALT_IN_ICE = cfg.initial_conditions_config.initial_ice_bulk_salinity

    BOTTOM_TEMP = cfg.initial_conditions_config.initial_ocean_temperature
    BOTTOM_SALT = 0
    TEMP_IN_ICE = cfg.initial_conditions_config.initial_ice_temperature

    INITIAL_OIL_VOLUME_FRACTION = (
        cfg.initial_conditions_config.initial_oil_volume_fraction
    )

    centers = Grids(cfg.numerical_params.I).centers
    salt = _apply_value_in_ice_layer(
        ICE_DEPTH, ice_value=SALT_IN_ICE, liquid_value=BOTTOM_SALT, grid=centers
    )
    gas = np.where(
        centers < -cfg.initial_conditions_config.initial_oil_free_depth,
        INITIAL_OIL_VOLUME_FRACTION,
        0,
    )

    temp = _apply_value_in_ice_layer(
        ICE_DEPTH, ice_value=TEMP_IN_ICE, liquid_value=BOTTOM_TEMP, grid=centers
    )
    solid_fraction_in_mush = (salt + temp) / (
        temp - cfg.physical_params.concentration_ratio
    )
    enthalpy = _apply_value_in_ice_layer(
        ICE_DEPTH,
        ice_value=temp - solid_fraction_in_mush * cfg.physical_params.stefan_number,
        liquid_value=temp,
        grid=centers,
    )

    return _pack_initial_state(cfg, enthalpy, salt, gas)


def _pack_initial_state(cfg: Config, enthalpy, salt, gas) -> State:
    match cfg.physical_params:
        case EQMPhysicalParams():
            return EQMState(0, enthalpy, salt, gas)
        case DISEQPhysicalParams():
            bulk_dissolved_gas = gas
            gas_fraction = np.zeros_like(gas)
            return DISEQState(0, enthalpy, salt, bulk_dissolved_gas, gas_fraction)
        case _:
            raise NotImplementedError
