from pathlib import Path
import numpy as np


from . import Config, DimensionalParams
from .params.physical import DISEQPhysicalParams, EQMPhysicalParams
from .state import get_unpacker
from .enthalpy_method import get_enthalpy_method


def load_data(
    sim_name: str,
    data_directory: Path,
    sim_config_name=None,
    is_dimensional=False,
    config_extension="yml",
):
    if sim_config_name is None:
        sim_config_name = sim_name

    SIM_DATA_PATH = data_directory / f"{sim_name}.npz"

    if is_dimensional:
        sim_cfg = DimensionalParams.load(
            data_directory / f"{sim_config_name}_dimensional.{config_extension}"
        ).get_config()
    else:
        sim_cfg = Config.load(data_directory / f"{sim_config_name}.{config_extension}")

    with np.load(SIM_DATA_PATH) as data:
        match sim_cfg.physical_params:
            case EQMPhysicalParams():
                times = data["arr_0"]
                enthalpy = data["arr_1"]
                salt = data["arr_2"]
                bulk_gas = data["arr_3"]

                data_tuple = (enthalpy, salt, bulk_gas)

            case DISEQPhysicalParams():
                times = data["arr_0"]
                enthalpy = data["arr_1"]
                salt = data["arr_2"]
                bulk_dissolved_gas = data["arr_3"]
                gas_fraction = data["arr_4"]

                data_tuple = (enthalpy, salt, bulk_dissolved_gas, gas_fraction)

            case _:
                raise NotImplementedError

    return sim_cfg, times, data_tuple


def get_state(non_dimensional_time, times, data, cfg):
    index = np.argmin(np.abs(times - non_dimensional_time))
    data_at_time = [quantity[:, index] for quantity in data]
    unpacker = get_unpacker(cfg)
    return unpacker(times[index], np.concatenate(tuple(data_at_time)))


def get_array_data(attr: str, cfg, times, data):
    data_slices = []
    for time in times:
        state = get_state(time, times, data, cfg)
        enthalpy_method = get_enthalpy_method(cfg)
        full_state = enthalpy_method(state)
        data_slices.append(getattr(full_state, attr))

    return np.vstack(tuple(data_slices)).T
