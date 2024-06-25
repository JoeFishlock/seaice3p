"""Module to compute the surface heat flux from geophysical energy balance

following [1]

Refs:
[1] P. D. Taylor and D. L. Feltham, ‘A model of melt pond evolution on sea ice’,
J. Geophys. Res., vol. 109, no. C12, p. 2004JC002361, Dec. 2004,
doi: 10.1029/2004JC002361.
"""
import numpy as np
from scipy.optimize import fsolve
from ..params import Config
from .turbulent_heat_flux import (
    calculate_latent_heat_flux,
    calculate_sensible_heat_flux,
)
from .radiative_forcing import get_LW_forcing, get_SW_forcing
from ..state.abstract_state import State

STEFAN_BOLTZMANN = 5.670374419e-8  # W/m2 K4
SW_ALBEDO = 0.7
SW_PENETRATION_FRACTION = 0.4


def calculate_emissivity(top_cell_is_ice: bool) -> float:
    if top_cell_is_ice:
        return 0.99
    return 0.97


def calculate_SW_penetration_fraction(cfg: Config, time: float) -> float:
    return SW_PENETRATION_FRACTION


def calculate_SW_albedo(cfg: Config, time: float) -> float:
    return SW_ALBEDO


def convert_surface_temperature_to_kelvin(
    cfg: Config, non_dimensional_surface_temperature: float
) -> float:
    surface_temperature_degrees_C = cfg.scales.convert_to_dimensional_temperature(
        non_dimensional_surface_temperature
    )
    return surface_temperature_degrees_C + 273.15


def calculate_total_heat_flux(
    cfg: Config, time: float, top_cell_is_ice: bool, surface_temp: float
) -> float:
    """Takes non-dimensional surface temperature and returns non-dimensional heat flux"""
    surface_temp_K = convert_surface_temperature_to_kelvin(cfg, surface_temp)
    emissivity = calculate_emissivity(top_cell_is_ice)
    SW_penetration_fraction = calculate_SW_penetration_fraction(cfg, time)
    SW_albedo = calculate_SW_albedo(cfg, time)
    dimensional_heat_flux = (
        get_LW_forcing(time, cfg)
        + (1 - SW_penetration_fraction) * (1 - SW_albedo) * get_SW_forcing(time, cfg)
        - emissivity * STEFAN_BOLTZMANN * surface_temp_K**4
        + calculate_sensible_heat_flux(cfg, time, top_cell_is_ice, surface_temp_K)
        + calculate_latent_heat_flux(cfg, time, top_cell_is_ice, surface_temp_K)
    )
    return cfg.scales.convert_from_dimensional_heat_flux(dimensional_heat_flux)


def top_cell_conductivity(cfg: Config, solid_fraction: float) -> float:
    return (
        1 - solid_fraction
    ) + solid_fraction * cfg.physical_params.conductivity_ratio


def surface_temp_gradient(
    cfg: Config,
    surface_temp: float,
    top_cell_center_temp: float,
    second_cell_center_temp: float,
) -> float:
    """Approximate non dimensional temperature gradient using the unknown surface
    temperature value (top of edge grid) and the top two known temperature values on
    the center grid
    """
    return (1 / (3 * cfg.numerical_params.step)) * (
        8 * surface_temp - 9 * top_cell_center_temp + second_cell_center_temp
    )


def solve_for_surface_temp(
    cfg: Config,
    time: float,
    top_cell_solid_fraction: float,
    top_cell_center_temp: float,
    second_cell_center_temp: float,
) -> float:
    """Returns non dimensional surface temperature"""
    if top_cell_solid_fraction == 0:
        top_cell_is_ice = False
    else:
        top_cell_is_ice = True

    def residual(surface_temperature: float) -> float:
        return top_cell_conductivity(
            cfg, top_cell_solid_fraction
        ) * surface_temp_gradient(
            cfg, surface_temperature, top_cell_center_temp, second_cell_center_temp
        ) - calculate_total_heat_flux(
            cfg, time, top_cell_is_ice, surface_temperature
        )

    initial_guess = top_cell_center_temp
    solution = fsolve(residual, initial_guess)[0]
    return solution


def find_ghost_cell_temperature(state: State) -> float:
    surface_temperature = solve_for_surface_temp(
        state.cfg,
        state.time,
        state.solid_fraction[-1],
        state.temperature[-1],
        state.temperature[-2],
    )
    return (
        state.cfg.numerical_params.step
        * surface_temp_gradient(
            state.cfg, surface_temp, state.temperature[-1], state.temperature[-2]
        )
        + state.temperature[-1]
    )
