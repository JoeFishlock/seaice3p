"""Calculate internal shortwave radiative heating due to oil droplets"""

import numpy as np
from oilrad import calculate_SW_heating_in_ice
from .state.abstract_state_bcs import StateBCs
from .RJW14.brine_drainage import calculate_ice_ocean_boundary_depth


def calculate_non_dimensional_shortwave_heating(state_bcs: StateBCs):
    """Calculate internal shortwave heating due to oil droplets on center grid"""
    # dimensional incident SW in W/m2
    INCIDENT_SW = 280
    MODEL_CHOICE = "1L"
    # To integrate spectrum between in nm
    MIN_WAVELENGTH = 350
    MAX_WAVELENGTH = 700

    center_grid = state_bcs.grid[1:-1]
    heating = np.zeros_like(center_grid)
    ice_ocean_boundary_depth = calculate_ice_ocean_boundary_depth(
        state_bcs.liquid_fraction, state_bcs.edge_grid
    )
    is_ice = center_grid > -ice_ocean_boundary_depth

    dimensional_ice_thickness = (
        ice_ocean_boundary_depth * state_bcs.cfg.scales.lengthscale
    )

    MODEL_KWARGS = {
        "oil_mass_ratio": 0,
        "ice_thickness": dimensional_ice_thickness,
        "ice_type": "FYI",
    }

    dimensional_heating = calculate_SW_heating_in_ice(
        INCIDENT_SW,
        center_grid[is_ice],
        MODEL_CHOICE,
        MIN_WAVELENGTH,
        MAX_WAVELENGTH,
        **MODEL_KWARGS
    )
    heating[is_ice] = state_bcs.cfg.scales.convert_from_dimensional_heating(
        dimensional_heating
    )
    return heating
