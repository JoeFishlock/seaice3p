"""Calculate internal shortwave radiative heating due to oil droplets"""

import numpy as np
from oilrad import calculate_SW_heating_in_ice
from ..grids import calculate_ice_ocean_boundary_depth
from .radiative_forcing import get_SW_forcing


def calculate_non_dimensional_shortwave_heating(state_bcs, cfg, grids):
    """Calculate internal shortwave heating due to oil droplets on center grid"""
    # To integrate spectrum between in nm
    MIN_WAVELENGTH = 350
    MAX_WAVELENGTH = 700

    center_grid = grids.centers
    edge_grid = grids.edges
    heating = np.zeros_like(center_grid)
    ice_ocean_boundary_depth = calculate_ice_ocean_boundary_depth(
        state_bcs.liquid_fraction, edge_grid
    )
    is_ice = center_grid > -ice_ocean_boundary_depth

    dimensional_ice_thickness = ice_ocean_boundary_depth * cfg.scales.lengthscale

    MODEL_KWARGS = {
        "oil_mass_ratio": cfg.forcing_config.constant_oil_mass_ratio,
        "ice_thickness": dimensional_ice_thickness,
        "ice_type": cfg.forcing_config.SW_scattering_ice_type,
    }

    dimensional_heating = calculate_SW_heating_in_ice(
        get_SW_forcing(state_bcs.time, cfg),
        center_grid[is_ice],
        cfg.forcing_config.SW_radiation_model_choice,
        MIN_WAVELENGTH,
        MAX_WAVELENGTH,
        **MODEL_KWARGS
    )
    heating[is_ice] = cfg.scales.convert_from_dimensional_heating(dimensional_heating)
    return heating
