import numpy as np


def get_temperature_forcing(time, params):
    choice = params.temperature_forcing_choice
    return TEMPERATURE_FORCINGS[choice](time, params)


def constant_temperature_forcing(time, params):
    return params.constant_top_temperature


def yearly_temperature_forcing(time, params):
    return 0.75 * (np.cos(time * 2 * np.pi / (4)) - 1)


TEMPERATURE_FORCINGS = {
    "constant": constant_temperature_forcing,
    "yearly": yearly_temperature_forcing,
}
