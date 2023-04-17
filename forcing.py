import numpy as np
from params import Config


def get_temperature_forcing(time, cfg: Config):
    choice = cfg.forcing_config.temperature_forcing_choice
    return TEMPERATURE_FORCINGS[choice](time, cfg)


def constant_temperature_forcing(time, cfg: Config):
    return cfg.forcing_config.constant_top_temperature


def yearly_temperature_forcing(time, cfg: Config):
    return 0.75 * (np.cos(time * 2 * np.pi / (4)) - 1)


TEMPERATURE_FORCINGS = {
    "constant": constant_temperature_forcing,
    "yearly": yearly_temperature_forcing,
}
