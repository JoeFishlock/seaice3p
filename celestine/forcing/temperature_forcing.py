"""Module for providing surface temperature forcing to simulation.

Note that the barrow temperature data is read in from a file if needed by the
simulation configuration.
"""
import numpy as np
from ..params import Config
from ..params.forcing import BRW09Forcing, YearlyForcing, ConstantForcing


def get_temperature_forcing(time, cfg: Config):
    TEMPERATURE_FORCINGS = {
        ConstantForcing: constant_temperature_forcing,
        YearlyForcing: yearly_temperature_forcing,
        BRW09Forcing: barrow_temperature_forcing,
    }
    return TEMPERATURE_FORCINGS[type(cfg.forcing_config)](time, cfg)


def get_bottom_temperature_forcing(time, cfg: Config):
    OCEAN_TEMPERATURE_FORCINGS = {
        ConstantForcing: constant_ocean_temperature_forcing,
        YearlyForcing: yearly_ocean_temperature_forcing,
        BRW09Forcing: barrow_ocean_temperature_forcing,
    }
    return OCEAN_TEMPERATURE_FORCINGS[type(cfg.forcing_config)](time, cfg)


def constant_temperature_forcing(time, cfg: Config):
    return cfg.forcing_config.constant_top_temperature


def yearly_temperature_forcing(time, cfg: Config):
    amplitude = cfg.forcing_config.amplitude
    period = cfg.forcing_config.period
    offset = cfg.forcing_config.offset
    return amplitude * (np.cos(time * 2 * np.pi / period) + offset)


def dimensional_barrow_temperature_forcing(time_in_days, cfg: Config):
    """Take time in days and linearly interp 2009 Barrow air/snow/ice temperature data to get
    temperature in degrees Celsius.
    """
    barrow_days = cfg.forcing_config.barrow_days
    barrow_top_temp = cfg.forcing_config.barrow_top_temp
    return np.interp(time_in_days, barrow_days, barrow_top_temp, right=np.NaN)


def barrow_temperature_forcing(time, cfg):
    """Take non dimensional time and return non dimensional air/snow/ice temperature at
    the Barrow site in 2009.

    For this to work you must have created the configuration cfg from dimensional
    parameters as it must have the conversion scales object.
    """
    time_in_days = cfg.scales.convert_to_dimensional_time(time)
    dimensional_temperature = dimensional_barrow_temperature_forcing(time_in_days, cfg)
    temperature = cfg.scales.convert_from_dimensional_temperature(
        dimensional_temperature
    )
    return temperature


def constant_ocean_temperature_forcing(time, cfg: Config):
    return cfg.forcing_config.ocean_temp


def yearly_ocean_temperature_forcing(time, cfg: Config):
    return cfg.forcing_config.ocean_temp


def dimensional_barrow_ocean_temperature_forcing(time_in_days, cfg: Config):
    """Take time in days and linearly interp 2009 Barrow ocean temperature data to get
    temperature in degrees Celsius.
    """
    barrow_ocean_days = cfg.forcing_config.barrow_ocean_days
    barrow_bottom_temp = cfg.forcing_config.barrow_bottom_temp
    return np.interp(time_in_days, barrow_ocean_days, barrow_bottom_temp, right=np.NaN)


def barrow_ocean_temperature_forcing(time, cfg):
    """Take non dimensional time and return non dimensional ocean temperature at
    the Barrow site in 2009.

    For this to work you must have created the configuration cfg from dimensional
    parameters as it must have the conversion scales object.
    """
    time_in_days = cfg.scales.convert_to_dimensional_time(time)
    dimensional_temperature = dimensional_barrow_ocean_temperature_forcing(
        time_in_days, cfg
    )
    temperature = cfg.scales.convert_from_dimensional_temperature(
        dimensional_temperature
    )
    return temperature
