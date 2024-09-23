from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


from . import (
    Config,
    DimensionalParams,
    get_config,
    DISEQPhysicalParams,
    EQMPhysicalParams,
    Grids,
)
from .state import EQMState, DISEQState, StateFull
from .enthalpy_method import get_enthalpy_method


@dataclass
class BaseResults:
    cfg: Config
    dcfg: None | DimensionalParams
    times: NDArray
    enthalpy: NDArray
    salt: NDArray


@dataclass
class EQMResults(BaseResults):
    bulk_gas: NDArray

    def __post_init__(self):
        _set_derived_attributes(
            self,
            [
                "solid_fraction",
                "liquid_fraction",
                "gas_fraction",
                "temperature",
                "liquid_salinity",
                "dissolved_gas",
            ],
        )


@dataclass
class DISEQResults(BaseResults):
    bulk_dissolved_gas: NDArray
    gas_fraction: NDArray

    def __post_init__(self):
        _set_derived_attributes(
            self,
            [
                "solid_fraction",
                "liquid_fraction",
                "temperature",
                "liquid_salinity",
                "dissolved_gas",
            ],
        )


Results = EQMResults | DISEQResults


def _set_derived_attributes(results: Results, enthalpy_method_attrs: list[str]) -> None:
    def get_state(time) -> StateFull:
        index = np.argmin(np.abs(results.times - time))
        match results:
            case EQMResults():
                state = EQMState(
                    results.times[index],
                    results.enthalpy[:, index],
                    results.salt[:, index],
                    results.bulk_gas[:, index],
                )
            case DISEQResults():
                state = DISEQState(
                    results.times[index],
                    results.enthalpy[:, index],
                    results.salt[:, index],
                    results.bulk_dissolved_gas[:, index],
                    results.gas_fraction[:, index],
                )

        enthalpy_method = get_enthalpy_method(results.cfg)
        return enthalpy_method(state)

    setattr(results, "states", list(map(get_state, results.times)))

    for attr in enthalpy_method_attrs:
        array_data = _get_array_data(attr, results.states)
        setattr(results, attr, array_data)

    setattr(results, "grids", Grids(results.cfg.numerical_params.I))


def load_simulation(
    sim_config_path: Path,
    sim_data_path: Path,
    is_dimensional: bool = True,
) -> Results:

    if is_dimensional:
        dcfg = DimensionalParams.load(sim_config_path)
        cfg = get_config(dcfg)
    else:
        dcfg = None
        cfg = Config.load(sim_config_path)

    with np.load(sim_data_path) as data:
        match cfg.physical_params:
            case EQMPhysicalParams():
                times = data["arr_0"]
                enthalpy = data["arr_1"]
                salt = data["arr_2"]
                bulk_gas = data["arr_3"]

                return EQMResults(cfg, dcfg, times, enthalpy, salt, bulk_gas)

            case DISEQPhysicalParams():
                times = data["arr_0"]
                enthalpy = data["arr_1"]
                salt = data["arr_2"]
                bulk_dissolved_gas = data["arr_3"]
                gas_fraction = data["arr_4"]

                return DISEQResults(
                    cfg, dcfg, times, enthalpy, salt, bulk_dissolved_gas, gas_fraction
                )

            case _:
                raise NotImplementedError


def _get_array_data(attr: str, states: list[StateFull]) -> NDArray:
    data_slices = []
    for state in states:
        data_slices.append(getattr(state, attr))

    return np.vstack(tuple(data_slices)).T
