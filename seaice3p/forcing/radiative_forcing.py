"""Module for providing surface radiative forcing to simulation.

Currently only total surface shortwave irradiance (integrated over entire shortwave
part of the spectrum) is provided and this is used to calculate internal radiative
heating.

Unlike temperature forcing this provides dimensional forcing
"""
from ..params import Config, RadForcing, ERA5Forcing
from ..state import StateBCs


def get_SW_penetration_fraction(state_bcs: StateBCs, cfg: Config) -> float:
    if isinstance(cfg.forcing_config, ERA5Forcing):
        if cfg.forcing_config.use_snow_data:
            # If there is snow cover no SW penetration
            if cfg.forcing_config.get_snow_depth(state_bcs.time) > 0:
                return 0

    # if there is ice set penetration through SSL
    if state_bcs.liquid_fraction[-2] < 1:
        return cfg.forcing_config.SW_forcing.SW_penetration_fraction
    return 1


def get_SW_forcing(time, cfg: Config):
    if isinstance(cfg.forcing_config, RadForcing):
        return _constant_SW_forcing(time, cfg)
    elif isinstance(cfg.forcing_config, ERA5Forcing):
        return cfg.forcing_config.get_SW(time)
    else:
        raise NotImplementedError


def _constant_SW_forcing(time, cfg: Config):
    """Returns constant surface shortwave downwelling irradiance in W/m2 integrated
    over the entire shortwave spectrum
    """
    return cfg.forcing_config.SW_forcing.SW_irradiance


def get_LW_forcing(time: float, cfg: Config) -> float:
    if isinstance(cfg.forcing_config, RadForcing):
        return _constant_LW_forcing(time, cfg)
    elif isinstance(cfg.forcing_config, ERA5Forcing):
        return cfg.forcing_config.get_LW(time)
    else:
        raise NotImplementedError


def _constant_LW_forcing(time, cfg: Config):
    """Returns constant surface longwave downwelling irradiance in W/m2 integrated
    over the entire longwave spectrum
    """
    return cfg.forcing_config.LW_forcing.LW_irradiance
