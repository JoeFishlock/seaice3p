"""Module for providing surface temperature forcing to simulation.

Note that the barrow temperature data is read in from a file if needed by the
simulation configuration.
"""
import numpy as np
from scipy.optimize import fsolve
from ..params import (
    Config,
    FixedTempOceanForcing,
    FixedHeatFluxOceanForcing,
    BRW09OceanForcing,
)
from ..params.forcing import (
    BRW09Forcing,
    YearlyForcing,
    ConstantForcing,
    RadForcing,
    ERA5Forcing,
    RobinForcing,
)
from .surface_energy_balance import find_ghost_cell_temperature
from ..state import StateFull
from ..equations.flux.heat_flux import calculate_conductivity


def get_temperature_forcing(state: StateFull, cfg: Config):
    TEMPERATURE_FORCINGS = {
        ConstantForcing: _constant_temperature_forcing,
        YearlyForcing: _yearly_temperature_forcing,
        BRW09Forcing: _barrow_temperature_forcing,
        RadForcing: find_ghost_cell_temperature,
        ERA5Forcing: find_ghost_cell_temperature,
        RobinForcing: _Robin_forcing,
    }
    return TEMPERATURE_FORCINGS[type(cfg.forcing_config)](state, cfg)


def get_bottom_temperature_forcing(state: StateFull, cfg: Config):
    OCEAN_TEMPERATURE_FORCINGS = {
        FixedTempOceanForcing: _constant_ocean_temperature_forcing,
        BRW09OceanForcing: _barrow_ocean_temperature_forcing,
        FixedHeatFluxOceanForcing: _constant_ocean_heat_flux_ghost_temperature,
    }
    return OCEAN_TEMPERATURE_FORCINGS[type(cfg.ocean_forcing_config)](state, cfg)


def _constant_temperature_forcing(state: StateFull, cfg: Config):
    return cfg.forcing_config.constant_top_temperature


def _yearly_temperature_forcing(state: StateFull, cfg: Config):
    amplitude = cfg.forcing_config.amplitude
    period = cfg.forcing_config.period
    offset = cfg.forcing_config.offset
    return amplitude * (np.cos(state.time * 2 * np.pi / period) + offset)


def _dimensional_barrow_temperature_forcing(time_in_days, cfg: Config):
    """Take time in days and linearly interp 2009 Barrow air/snow/ice temperature data to get
    temperature in degrees Celsius.
    """
    barrow_days = cfg.forcing_config.barrow_days
    barrow_top_temp = cfg.forcing_config.barrow_top_temp
    return np.interp(time_in_days, barrow_days, barrow_top_temp, right=np.NaN)


def _barrow_temperature_forcing(state: StateFull, cfg: Config):
    """Take non dimensional time and return non dimensional air/snow/ice temperature at
    the Barrow site in 2009.

    For this to work you must have created the configuration cfg from dimensional
    parameters as it must have the conversion scales object.
    """
    time_in_days = cfg.scales.convert_to_dimensional_time(state.time)
    dimensional_temperature = _dimensional_barrow_temperature_forcing(time_in_days, cfg)
    temperature = cfg.scales.convert_from_dimensional_temperature(
        dimensional_temperature
    )
    return temperature


def _Robin_forcing(state: StateFull, cfg: Config):
    """Returns non dimensional ghost cell temperature such that surface heat flux
    is given by Robin boundary condition"""

    def residual(ghost_cell_temperature: float) -> float:
        surface_temperature = 0.5 * (ghost_cell_temperature + state.temperature[-1])
        temp_gradient = (1 / cfg.numerical_params.step) * (
            ghost_cell_temperature - state.temperature[-1]
        )
        return calculate_conductivity(
            cfg, state.solid_fraction[-1]
        ) * temp_gradient - cfg.forcing_config.biot * (
            cfg.forcing_config.restoring_temperature - surface_temperature
        )

    initial_guess = state.temperature[-1]
    solution = fsolve(residual, initial_guess)[0]
    return solution


def _constant_ocean_temperature_forcing(state: StateFull, cfg: Config) -> float:
    return cfg.ocean_forcing_config.ocean_temp


def _dimensional_barrow_ocean_temperature_forcing(
    time_in_days: float, cfg: Config
) -> float:
    """Take time in days and linearly interp 2009 Barrow ocean temperature data to get
    temperature in degrees Celsius.
    """
    barrow_ocean_days = cfg.ocean_forcing_config.barrow_ocean_days
    barrow_bottom_temp = cfg.ocean_forcing_config.barrow_bottom_temp
    return np.interp(time_in_days, barrow_ocean_days, barrow_bottom_temp, right=np.NaN)


def _barrow_ocean_temperature_forcing(state: StateFull, cfg: Config) -> float:
    """Take non dimensional time and return non dimensional ocean temperature at
    the Barrow site in 2009.

    For this to work you must have created the configuration cfg from dimensional
    parameters as it must have the conversion scales object.
    """
    time_in_days = cfg.scales.convert_to_dimensional_time(state.time)
    dimensional_temperature = _dimensional_barrow_ocean_temperature_forcing(
        time_in_days, cfg
    )
    temperature = cfg.scales.convert_from_dimensional_temperature(
        dimensional_temperature
    )
    return temperature


def _constant_ocean_heat_flux_ghost_temperature(state: StateFull, cfg: Config) -> float:
    conductivity = calculate_conductivity(cfg, state.solid_fraction[0])
    return state.temperature[0] + (
        (cfg.ocean_forcing_config.ocean_heat_flux * cfg.numerical_params.step)
        / conductivity
    )
