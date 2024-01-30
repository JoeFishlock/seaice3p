"""Module to provide functions to add boundary conditions to each quantity on the
centered grid that needs to be on the ghost grid for the upwind scheme.
"""

from celestine.forcing import get_temperature_forcing, get_bottom_temperature_forcing
from celestine.grids import add_ghost_cells
from celestine.params import Config


def dissolved_gas_BCs(dissolved_gas_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    return add_ghost_cells(
        dissolved_gas_centers, bottom=cfg.boundary_conditions_config.far_gas_sat, top=1
    )


def gas_fraction_BCs(gas_fraction_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    return add_ghost_cells(gas_fraction_centers, bottom=0, top=0)


def gas_BCs(gas_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    chi = cfg.physical_params.expansion_coefficient
    far_gas_sat = cfg.boundary_conditions_config.far_gas_sat
    return add_ghost_cells(gas_centers, bottom=chi * far_gas_sat, top=chi)


def liquid_salinity_BCs(liquid_salinity_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    far_bulk_salt = cfg.boundary_conditions_config.far_bulk_salinity
    return add_ghost_cells(
        liquid_salinity_centers, bottom=far_bulk_salt, top=liquid_salinity_centers[-1]
    )


def temperature_BCs(temperature_centers, time, cfg: Config):
    """Add ghost cells with BCs to center quantity

    Note this needs the current time as well as top temperature is forced."""
    far_temp = get_bottom_temperature_forcing(time, cfg)
    top_temp = get_temperature_forcing(time, cfg)
    return add_ghost_cells(temperature_centers, bottom=far_temp, top=top_temp)


def enthalpy_BCs(enthalpy_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    far_temp = cfg.boundary_conditions_config.far_temp
    return add_ghost_cells(enthalpy_centers, bottom=far_temp, top=enthalpy_centers[-1])


def salt_BCs(salt_centers, cfg: Config):
    """Add ghost cells with BCs to center quantity"""
    far_bulk_salt = cfg.boundary_conditions_config.far_bulk_salinity
    return add_ghost_cells(salt_centers, bottom=far_bulk_salt, top=salt_centers[-1])


def liquid_fraction_BCs(liquid_fraction_centers, cfg: Config):
    """Add ghost cells to liquid fraction such that top and bottom boundaries take the
    same value as the top and bottom cell center"""
    return add_ghost_cells(
        liquid_fraction_centers,
        bottom=liquid_fraction_centers[0],
        top=liquid_fraction_centers[-1],
    )


def pressure_BCs(pressure_centers, cfg: Config):
    """Add ghost cells to pressure so that W_l=0 at z=0 and p=0 at z=-1"""
    return add_ghost_cells(pressure_centers, bottom=0, top=pressure_centers[-1])
