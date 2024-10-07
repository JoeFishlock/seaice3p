"""Module to provide functions to add boundary conditions to each quantity on the
centered grid that needs to be on the ghost grid for the upwind scheme.
"""

from typing import Callable
import numpy as np

from .temperature_forcing import get_temperature_forcing, get_bottom_temperature_forcing
from ..grids import add_ghost_cells
from ..params import (
    Config,
    EQMPhysicalParams,
    DISEQPhysicalParams,
    OilInitialConditions,
)
from ..state import (
    StateFull,
    StateBCs,
    EQMStateFull,
    DISEQStateFull,
    EQMStateBCs,
    DISEQStateBCs,
)


def get_boundary_conditions(cfg: Config) -> Callable[[StateFull], StateBCs]:
    fun_map = {
        EQMPhysicalParams: _EQM_boundary_conditions,
        DISEQPhysicalParams: _DISEQ_boundary_conditions,
    }

    def boundary_conditions(full_state: StateFull) -> StateBCs:
        return fun_map[type(cfg.physical_params)](full_state, cfg)

    return boundary_conditions


def _EQM_boundary_conditions(full_state: EQMStateFull, cfg: Config) -> StateBCs:
    time = full_state.time
    temperature = _temperature_BCs(full_state, cfg)
    enthalpy = _enthalpy_BCs(full_state.enthalpy, cfg, temperature[0])
    salt = _salt_BCs(full_state.salt, cfg)

    liquid_salinity = _liquid_salinity_BCs(full_state.liquid_salinity, cfg)
    dissolved_gas = _dissolved_gas_BCs(full_state.dissolved_gas, cfg)
    gas_fraction = _gas_fraction_BCs(full_state.gas_fraction, cfg)
    liquid_fraction = _liquid_fraction_BCs(full_state.liquid_fraction)

    gas = _gas_BCs(full_state.gas, cfg)
    return EQMStateBCs(
        time,
        enthalpy,
        salt,
        gas,
        temperature,
        liquid_salinity,
        dissolved_gas,
        gas_fraction,
        liquid_fraction,
    )


def _DISEQ_boundary_conditions(full_state: DISEQStateFull, cfg: Config) -> StateBCs:
    time = full_state.time
    temperature = _temperature_BCs(full_state, cfg)
    enthalpy = _enthalpy_BCs(full_state.enthalpy, cfg, temperature[0])
    salt = _salt_BCs(full_state.salt, cfg)

    liquid_salinity = _liquid_salinity_BCs(full_state.liquid_salinity, cfg)
    dissolved_gas = _dissolved_gas_BCs(full_state.dissolved_gas, cfg)
    gas_fraction = _gas_fraction_BCs(full_state.gas_fraction, cfg)
    liquid_fraction = _liquid_fraction_BCs(full_state.liquid_fraction)

    bulk_dissolved_gas = (
        cfg.physical_params.expansion_coefficient * liquid_fraction * dissolved_gas
    )
    return DISEQStateBCs(
        time,
        enthalpy,
        salt,
        temperature,
        liquid_salinity,
        dissolved_gas,
        liquid_fraction,
        bulk_dissolved_gas,
        gas_fraction,
    )


def _dissolved_gas_BCs(dissolved_gas_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    return add_ghost_cells(
        dissolved_gas_centers, bottom=cfg.ocean_forcing_config.ocean_gas_sat, top=1
    )


def _gas_fraction_BCs(gas_fraction_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    if isinstance(cfg.initial_conditions_config, OilInitialConditions):
        return add_ghost_cells(
            gas_fraction_centers,
            bottom=cfg.initial_conditions_config.initial_oil_volume_fraction,
            top=0,
        )
    else:
        return add_ghost_cells(
            gas_fraction_centers, bottom=gas_fraction_centers[0], top=0
        )


def _gas_BCs(gas_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    chi = cfg.physical_params.expansion_coefficient
    far_gas_sat = cfg.ocean_forcing_config.ocean_gas_sat
    return add_ghost_cells(gas_centers, bottom=chi * far_gas_sat, top=chi)


def _liquid_salinity_BCs(liquid_salinity_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    return add_ghost_cells(
        liquid_salinity_centers, bottom=0, top=liquid_salinity_centers[-1]
    )


def _temperature_BCs(state: StateFull, cfg: Config):
    """Add ghost cells with BCs to center quantity

    Note this needs the current time as well as top temperature is forced."""
    far_temp = get_bottom_temperature_forcing(state, cfg)
    top_temp = get_temperature_forcing(state, cfg)
    return add_ghost_cells(state.temperature, bottom=far_temp, top=top_temp)


def _enthalpy_BCs(enthalpy_centers, cfg: Config, bottom_temperature):
    """Add ghost cells with BCs to center quantity"""
    return add_ghost_cells(
        enthalpy_centers, bottom=bottom_temperature, top=enthalpy_centers[-1]
    )


def _salt_BCs(salt_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    return add_ghost_cells(salt_centers, bottom=0, top=salt_centers[-1])


def _liquid_fraction_BCs(liquid_fraction_centers):
    """Add ghost cells to liquid fraction such that top and bottom boundaries take the
    same value as the top and bottom cell center"""
    return add_ghost_cells(
        liquid_fraction_centers,
        bottom=liquid_fraction_centers[0],
        top=liquid_fraction_centers[-1],
    )
