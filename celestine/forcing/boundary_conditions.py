"""Module to provide functions to add boundary conditions to each quantity on the
centered grid that needs to be on the ghost grid for the upwind scheme.
"""

from typing import Callable

from .temperature_forcing import get_temperature_forcing, get_bottom_temperature_forcing
from .surface_energy_balance import find_ghost_cell_temperature
from ..grids import add_ghost_cells
from ..params import Config
from ..params.forcing import RadForcing
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
        "EQM": _EQM_boundary_conditions,
        "DISEQ": _DISEQ_boundary_conditions,
    }

    def boundary_conditions(full_state: StateFull) -> StateBCs:
        return fun_map[cfg.model](full_state, cfg)

    return boundary_conditions


def _EQM_boundary_conditions(full_state: EQMStateFull, cfg: Config) -> StateBCs:
    time = full_state.time
    enthalpy = _enthalpy_BCs(full_state.enthalpy, cfg)
    salt = _salt_BCs(full_state.salt, cfg)

    temperature = _temperature_BCs(full_state, time, cfg)
    liquid_salinity = _liquid_salinity_BCs(full_state.liquid_salinity, cfg)
    dissolved_gas = _dissolved_gas_BCs(full_state.dissolved_gas, cfg)
    gas_fraction = _gas_fraction_BCs(full_state.gas_fraction)
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
    enthalpy = _enthalpy_BCs(full_state.enthalpy, cfg)
    salt = _salt_BCs(full_state.salt, cfg)

    temperature = _temperature_BCs(full_state, time, cfg)
    liquid_salinity = _liquid_salinity_BCs(full_state.liquid_salinity, cfg)
    dissolved_gas = _dissolved_gas_BCs(full_state.dissolved_gas, cfg)
    gas_fraction = _gas_fraction_BCs(full_state.gas_fraction)
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
        dissolved_gas_centers, bottom=cfg.forcing_config.ocean_gas_sat, top=1
    )


def _gas_fraction_BCs(gas_fraction_centers):
    """Add ghost cells with BCs to center quantity"""
    return add_ghost_cells(gas_fraction_centers, bottom=0, top=0)


def _gas_BCs(gas_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    chi = cfg.physical_params.expansion_coefficient
    far_gas_sat = cfg.forcing_config.ocean_gas_sat
    return add_ghost_cells(gas_centers, bottom=chi * far_gas_sat, top=chi)


def _liquid_salinity_BCs(liquid_salinity_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    far_bulk_salt = cfg.forcing_config.ocean_bulk_salinity
    return add_ghost_cells(
        liquid_salinity_centers, bottom=far_bulk_salt, top=liquid_salinity_centers[-1]
    )


def _temperature_BCs(state, time, cfg: Config):
    """Add ghost cells with BCs to center quantity

    Note this needs the current time as well as top temperature is forced."""
    far_temp = get_bottom_temperature_forcing(time, cfg)

    if isinstance(cfg.forcing_config, RadForcing):
        return add_ghost_cells(
            state.temperature, bottom=far_temp, top=find_ghost_cell_temperature(state)
        )

    top_temp = get_temperature_forcing(time, cfg)
    return add_ghost_cells(state.temperature, bottom=far_temp, top=top_temp)


def _enthalpy_BCs(enthalpy_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    far_temp = cfg.forcing_config.ocean_temp
    return add_ghost_cells(enthalpy_centers, bottom=far_temp, top=enthalpy_centers[-1])


def _salt_BCs(salt_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    far_bulk_salt = cfg.forcing_config.ocean_bulk_salinity
    return add_ghost_cells(salt_centers, bottom=far_bulk_salt, top=salt_centers[-1])


def _liquid_fraction_BCs(liquid_fraction_centers):
    """Add ghost cells to liquid fraction such that top and bottom boundaries take the
    same value as the top and bottom cell center"""
    return add_ghost_cells(
        liquid_fraction_centers,
        bottom=liquid_fraction_centers[0],
        top=liquid_fraction_centers[-1],
    )
