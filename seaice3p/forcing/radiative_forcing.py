"""Module for providing surface radiative forcing to simulation.

Currently only total surface shortwave irradiance (integrated over entire shortwave
part of the spectrum) is provided and this is used to calculate internal radiative
heating.

Unlike temperature forcing this provides dimensional forcing
"""
from ..params import Config
from ..params.dimensional import (
    DimensionalConstantSWForcing,
    DimensionalConstantLWForcing,
)
from ..state import StateBCs


def get_SW_penetration_fraction(state_bcs: StateBCs, cfg: Config) -> float:
    # if there is ice set penetration through SSL
    if state_bcs.liquid_fraction[-2] < 1:
        return cfg.forcing_config.SW_forcing.SW_penetration_fraction
    return 1


def get_SW_forcing(time, cfg: Config):
    SW_FORCINGS = {
        DimensionalConstantSWForcing: _constant_SW_forcing,
    }
    return SW_FORCINGS[type(cfg.forcing_config.SW_forcing)](time, cfg)


def _constant_SW_forcing(time, cfg: Config):
    """Returns constant surface shortwave downwelling irradiance in W/m2 integrated
    over the entire shortwave spectrum
    """
    return cfg.forcing_config.SW_forcing.SW_irradiance


def get_LW_forcing(time: float, cfg: Config) -> float:
    LW_FORCINGS = {
        DimensionalConstantLWForcing: _constant_LW_forcing,
    }
    return LW_FORCINGS[type(cfg.forcing_config.LW_forcing)](time, cfg)


def _constant_LW_forcing(time, cfg: Config):
    """Returns constant surface longwave downwelling irradiance in W/m2 integrated
    over the entire longwave spectrum
    """
    return cfg.forcing_config.LW_forcing.LW_irradiance
