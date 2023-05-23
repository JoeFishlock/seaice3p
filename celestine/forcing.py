import numpy as np
from celestine.params import Config

"""indices of each data variable for Barrow 2009 data are in the metadata
celestine/forcing_data/BRW09_MBS_metadata.txt
"""
AIR_TEMP_INDEX = 8
TIME_INDEX = 0
BARROW_DATA_PATH = "celestine/forcing_data/BRW09.txt"


def filter_missing_values(air_temp, days):
    """Filter out missing values are recorded as 9999"""
    is_missing = np.abs(air_temp) > 100
    return air_temp[~is_missing], days[~is_missing]


def read_barrow_data(path):
    data = np.genfromtxt(path, delimiter="\t")
    air_temp = data[:, AIR_TEMP_INDEX]
    days = data[:, TIME_INDEX] - data[0, TIME_INDEX]
    air_temp, days = filter_missing_values(air_temp, days)
    return air_temp, days


AIR_TEMP, DAYS = read_barrow_data(BARROW_DATA_PATH)


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


def dimensional_barrow_temperature_forcing(time_in_days):
    """Take time in days and linearly interp 2009 Barrow air temperature data to get
    temperature in degrees Celsius.
    """
    return np.interp(time_in_days, DAYS, AIR_TEMP, right=np.NaN)


def barrow_temperature_forcing(time, cfg):
    """Take non dimensional time and return non dimensional air temperature at
    the Barrow site in 2009.

    For this to work you must have created the configuration cfg from dimensional
    parameters as it must have the conversion scales object.
    """
    time_in_days = cfg.scales.convert_to_dimensional_time(time)
    dimensional_temperature = dimensional_barrow_temperature_forcing(time_in_days)
    temperature = cfg.scales.convert_from_dimensional_temperature(
        dimensional_temperature
    )
    return temperature


TEMPERATURE_FORCINGS = {
    "constant": constant_temperature_forcing,
    "yearly": yearly_temperature_forcing,
    "barrow_2009": barrow_temperature_forcing,
}
