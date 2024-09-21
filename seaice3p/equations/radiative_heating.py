"""Calculate internal shortwave radiative heating due to oil droplets"""

from typing import Callable
import numpy as np
from numpy.typing import NDArray
import oilrad as oi
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
    state_bcs: StateBCs, cfg: Config, grids: Grids
):
    """Calculate internal shortwave heating due to oil droplets on center grid

    Assumes a configuration with the RadForcing object as the forcing config is
    passed."""

    center_grid = grids.centers
    heating = np.zeros_like(center_grid)

    # If we don't have radiative forcing then just return array of zeros for heating
    if not isinstance(cfg.forcing_config, RadForcing):
        return heating

    SW_RANGE = (
        cfg.forcing_config.SW_forcing.SW_min_wavelength,
        cfg.forcing_config.SW_forcing.SW_max_wavelength,
    )
    NUM_WAVELENGTH_SAMPLES = cfg.forcing_config.SW_forcing.num_wavelength_samples
    MEDIAN_DROPLET_RADIUS_MICRONS = (
        cfg.scales.pore_radius * cfg.bubble_params.bubble_radius_scaled * 1e6
    )
    OIL_DENSITY = 900
    ICE_DENSITY = 916
    wavelengths = np.geomspace(SW_RANGE[0], SW_RANGE[1], NUM_WAVELENGTH_SAMPLES)
    convert_gas_fraction_to_oil_mass = lambda phi: phi * 1e9 * OIL_DENSITY / ICE_DENSITY

    edge_grid = grids.edges

    def interp_centers_to_edges(
        center_quantity,
    ):
        return np.interp(edge_grid, center_grid, center_quantity)

    match cfg.forcing_config.oil_heating:
        case DimensionalNoHeating():
            return heating
        case DimensionalBackgroundOilHeating():
            oil_mass_ratio = np.full_like(
                edge_grid, cfg.forcing_config.oil_heating.oil_mass_ratio
            )

        case DimensionalMobileOilHeating():
            gas_fraction_centers = state_bcs.gas_fraction[1:-1]
            gas_fraction_edges = interp_centers_to_edges(gas_fraction_centers)
            oil_mass_ratio = convert_gas_fraction_to_oil_mass(gas_fraction_edges)
        case _:
            raise NotImplementedError()

    center_liquid_fraction = state_bcs.liquid_fraction[1:-1]
    model = oi.InfiniteLayerModel(
        edge_grid,
        wavelengths,
        oil_mass_ratio=oil_mass_ratio,
        ice_type=cfg.forcing_config.oil_heating.ice_type,
        median_droplet_radius_in_microns=MEDIAN_DROPLET_RADIUS_MICRONS,
        liquid_fraction=interp_centers_to_edges(center_liquid_fraction),
    )
    irradiances = oi.integrate_over_SW(oi.solve_two_stream_model(model))

    # if there is ice set penetration through SSL
    if center_liquid_fraction[-1] < 1:
        PEN = cfg.forcing_config.SW_forcing.SW_penetration_fraction
    else:
        PEN = 1

    print(PEN)
    # dimensional heating rate in W/m3
    heating = (
        PEN
        * get_SW_forcing(state_bcs.time, cfg)
        * np.diff(irradiances.net_irradiance)
        / np.diff(edge_grid * cfg.scales.lengthscale)
    )

    return cfg.scales.convert_from_dimensional_heating(heating)
