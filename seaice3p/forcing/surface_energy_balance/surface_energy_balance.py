"""Module to compute the surface heat flux from geophysical energy balance

following [1]

Refs:
[1] P. D. Taylor and D. L. Feltham, ‘A model of melt pond evolution on sea ice’,
J. Geophys. Res., vol. 109, no. C12, p. 2004JC002361, Dec. 2004,
doi: 10.1029/2004JC002361.
"""
from scipy.optimize import fsolve
from ...state import StateFull
from ...params import Config, ERA5Forcing
from .turbulent_heat_flux import (
    calculate_latent_heat_flux,
    calculate_sensible_heat_flux,
)
from ..radiative_forcing import get_LW_forcing
from ...equations.flux.heat_flux import calculate_conductivity

STEFAN_BOLTZMANN = 5.670374419e-8  # W/m2 K4


def _calculate_emissivity(cfg: Config, top_cell_is_ice: bool) -> float:
    if top_cell_is_ice:
        return cfg.forcing_config.LW_forcing.ice_emissitivty
    return cfg.forcing_config.LW_forcing.water_emissivity


def _convert_non_dim_temperature_to_kelvin(
    cfg: Config, non_dimensional_surface_temperature: float
) -> float:
    surface_temperature_degrees_C = cfg.scales.convert_to_dimensional_temperature(
        non_dimensional_surface_temperature
    )
    return surface_temperature_degrees_C + 273.15


def _calculate_total_heat_flux(
    cfg: Config,
    time: float,
    top_cell_is_ice: bool,
    surface_temp: float,
    temp_gradient: float,
    top_cell_conductivity: float,
) -> float:
    """Takes non-dimensional surface temperature and returns non-dimensional heat flux"""
    if isinstance(cfg.forcing_config, ERA5Forcing) and cfg.forcing_config.use_snow_data:
        dimensional_temperature_gradient = (
            cfg.scales.temperature_difference * temp_gradient / cfg.scales.lengthscale
        )
        surface_temp_K = (
            _convert_non_dim_temperature_to_kelvin(cfg, surface_temp)
            + (top_cell_conductivity / cfg.physical_params.snow_conductivity_ratio)
            * cfg.forcing_config.get_snow_depth(time)
            * dimensional_temperature_gradient
        )
    else:
        surface_temp_K = _convert_non_dim_temperature_to_kelvin(cfg, surface_temp)
    emissivity = _calculate_emissivity(cfg, top_cell_is_ice)
    dimensional_heat_flux = (
        get_LW_forcing(time, cfg)
        - emissivity * STEFAN_BOLTZMANN * surface_temp_K**4
        + calculate_sensible_heat_flux(cfg, time, top_cell_is_ice, surface_temp_K)
        + calculate_latent_heat_flux(cfg, time, top_cell_is_ice, surface_temp_K)
    )
    return cfg.scales.convert_from_dimensional_heat_flux(dimensional_heat_flux)


def find_ghost_cell_temperature(state: StateFull, cfg: Config) -> float:
    """Returns non dimensional ghost cell temperature such that surface heat flux
    is the sum of incoming LW, outgoing LW, sensible and latent heat fluxes.
    The SW heat flux is determined in the radiative heating term."""
    if state.solid_fraction[-1] == 0:
        top_cell_is_ice = False
    else:
        top_cell_is_ice = True

    def residual(ghost_cell_temperature: float) -> float:
        surface_temperature = 0.5 * (ghost_cell_temperature + state.temperature[-1])
        temp_gradient = (1 / cfg.numerical_params.step) * (
            ghost_cell_temperature - state.temperature[-1]
        )
        conductivity = calculate_conductivity(cfg, state.solid_fraction[-1])
        return conductivity * temp_gradient - _calculate_total_heat_flux(
            cfg,
            state.time,
            top_cell_is_ice,
            surface_temperature,
            temp_gradient,
            conductivity,
        )

    initial_guess = state.temperature[-1]
    solution = fsolve(residual, initial_guess)[0]
    return solution
