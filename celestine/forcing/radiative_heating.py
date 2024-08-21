"""Calculate internal shortwave radiative heating due to oil droplets"""

import numpy as np
from oilrad import calculate_SW_heating_in_ice
from ..RJW14.brine_drainage import calculate_ice_ocean_boundary_depth
from .radiative_forcing import get_SW_forcing


def calculate_non_dimensional_shortwave_heating(state_bcs):
    """Calculate internal shortwave heating due to oil droplets on center grid"""
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
        "oil_mass_ratio": state_bcs.cfg.forcing_config.constant_oil_mass_ratio,
        "ice_thickness": dimensional_ice_thickness,
        "ice_type": state_bcs.cfg.forcing_config.SW_scattering_ice_type,
    }

    dimensional_heating = calculate_SW_heating_in_ice(
        get_SW_forcing(state_bcs.time, state_bcs.cfg),
        center_grid[is_ice],
        state_bcs.cfg.forcing_config.SW_radiation_model_choice,
        MIN_WAVELENGTH,
        MAX_WAVELENGTH,
        **MODEL_KWARGS
    )
    heating[is_ice] = state_bcs.cfg.scales.convert_from_dimensional_heating(
        dimensional_heating
    )
    return heating
