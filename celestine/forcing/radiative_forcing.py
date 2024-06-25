"""Module for providing surface radiative forcing to simulation.

Currently only total surface shortwave irradiance (integrated over entire shortwave
part of the spectrum) is provided and this is used to calculate internal radiative
heating.

Unlike temperature forcing this provides dimensional forcing
"""
from ..params import Config

LW_IRRADIANCE = 260  # W/m2


def get_SW_forcing(time, cfg: Config):
    SW_FORCINGS = {
        "constant": constant_SW_forcing,
    }
    choice = cfg.forcing_config.SW_forcing_choice
    return SW_FORCINGS[choice](time, cfg)


def constant_SW_forcing(time, cfg: Config):
    """Returns constant surface shortwave downwelling irradiance in W/m2 integrated
    over the entire shortwave spectrum
    """
    return cfg.forcing_config.constant_SW_irradiance


def get_LW_forcing(time: float, cfg: Config) -> float:
    return LW_IRRADIANCE
