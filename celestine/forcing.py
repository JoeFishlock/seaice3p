import numpy as np
from celestine.params import Config


def get_temperature_forcing(time, cfg: Config):
    choice = cfg.forcing_config.temperature_forcing_choice
    return TEMPERATURE_FORCINGS[choice](time, cfg)


def constant_temperature_forcing(time, cfg: Config):
    return cfg.forcing_config.constant_top_temperature


def yearly_temperature_forcing(time, cfg: Config):
    amplitude = cfg.forcing_config.amplitude
    period = cfg.forcing_config.period
    offset = cfg.forcing_config.offset
    return amplitude * (np.cos(time * 2 * np.pi / period) + offset)


TEMPERATURE_FORCINGS = {
    "constant": constant_temperature_forcing,
    "yearly": yearly_temperature_forcing,
}
