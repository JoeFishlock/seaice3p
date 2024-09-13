"""Calculate internal shortwave radiative heating due to oil droplets"""

from typing import Callable
import numpy as np
from numpy.typing import NDArray
from oilrad import calculate_SW_heating_in_ice
from ..grids import calculate_ice_ocean_boundary_depth, Grids
from ..params import Config, EQMPhysicalParams, DISEQPhysicalParams, RadForcing
from ..params.dimensional import (
    DimensionalBackgroundOilHeating,
    DimensionalMobileOilHeating,
    DimensionalNoHeating,
)
from ..forcing import get_SW_forcing
from ..state import StateBCs, EQMStateBCs, DISEQStateBCs


def get_radiative_heating(cfg: Config, grids: Grids) -> Callable[[StateBCs], NDArray]:
    """Calculate internal shortwave heating source for enthalpy equation.

    if the RadForcing object is given as the forcing config then calculates internal
    heating based on the object given in the configuration for oil_heating.

    If another forcing is chosen then just returns a function to create an array of
    zeros as no internal heating is calculated.
    """
    fun_map = {
        EQMPhysicalParams: _EQM_radiative_heating,
        DISEQPhysicalParams: _DISEQ_radiative_heating,
    }

    def radiative_heating(state_BCs: StateBCs) -> NDArray:
        return fun_map[type(cfg.physical_params)](state_BCs, cfg, grids)

    return radiative_heating


def _EQM_radiative_heating(
    state_BCs: EQMStateBCs, cfg: Config, grids: Grids
) -> NDArray:
    heating = _calculate_non_dimensional_shortwave_heating(state_BCs, cfg, grids)
    return np.hstack(
        (
            heating,
            np.zeros_like(heating),
            np.zeros_like(heating),
        )
    )


def _DISEQ_radiative_heating(
    state_BCs: DISEQStateBCs, cfg: Config, grids: Grids
) -> NDArray:
    heating = _calculate_non_dimensional_shortwave_heating(state_BCs, cfg, grids)
    return np.hstack(
        (
            heating,
            np.zeros_like(heating),
            np.zeros_like(heating),
            np.zeros_like(heating),
        )
    )


def _calculate_non_dimensional_shortwave_heating(
    state_bcs: StateBCs, cfg: Config, grids
):
    """Calculate internal shortwave heating due to oil droplets on center grid

    Assumes a configuration with the RadForcing object as the forcing config is
    passed."""
    # To integrate spectrum between in nm
    MIN_WAVELENGTH = 350
    MAX_WAVELENGTH = 700

    center_grid = grids.centers
    edge_grid = grids.edges
    heating = np.zeros_like(center_grid)

    # If we don't have radiative forcing then just return array of zeros for heating
    if not isinstance(cfg.forcing_config, RadForcing):
        return heating

    ice_ocean_boundary_depth = calculate_ice_ocean_boundary_depth(
        state_bcs.liquid_fraction, edge_grid
    )
    is_ice = center_grid > -ice_ocean_boundary_depth

    dimensional_ice_thickness = ice_ocean_boundary_depth * cfg.scales.lengthscale

    match cfg.forcing_config.oil_heating:
        case DimensionalNoHeating():
            return heating
        case DimensionalBackgroundOilHeating():
            MODEL_KWARGS = {
                "oil_mass_ratio": cfg.forcing_config.oil_heating.oil_mass_ratio,
                "ice_thickness": dimensional_ice_thickness,
                "ice_type": cfg.forcing_config.oil_heating.ice_type,
            }
            model_choice = "1L"

        case DimensionalMobileOilHeating():
            oil_mass_ratio = lambda z: np.interp(
                z, center_grid, state_bcs.gas_fraction[1:-1] * 0.9 * 1e9
            )
            MODEL_KWARGS = {
                "oil_mass_ratio": oil_mass_ratio,
                "ice_thickness": dimensional_ice_thickness,
                "ice_type": cfg.forcing_config.oil_heating.ice_type,
            }
            model_choice = "IL"
        case _:
            raise NotImplementedError()

    dimensional_heating = calculate_SW_heating_in_ice(
        get_SW_forcing(state_bcs.time, cfg),
        center_grid[is_ice],
        model_choice,
        MIN_WAVELENGTH,
        MAX_WAVELENGTH,
        **MODEL_KWARGS
    )

    heating[is_ice] = cfg.scales.convert_from_dimensional_heating(dimensional_heating)
    return heating
