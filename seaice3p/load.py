from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import oilrad as oi


from . import (
    Config,
    DimensionalParams,
    get_config,
    DISEQPhysicalParams,
    EQMPhysicalParams,
    Grids,
    RadForcing,
)
from .state import EQMState, DISEQState, DISEQStateFull, EQMStateFull, StateFull
from .enthalpy_method import get_enthalpy_method
from .forcing.boundary_conditions import get_boundary_conditions
from .forcing import get_SW_penetration_fraction
from .equations.radiative_heating import run_two_stream_model


@dataclass
class _BaseResults:
    cfg: Config
    dcfg: None | DimensionalParams
    times: NDArray
    enthalpy: NDArray
    salt: NDArray

    def __post_init__(self):
        self.states = list(map(self._get_state, self.times))

        boundary_conditions = get_boundary_conditions(self.cfg)
        self.states_bcs = list(map(boundary_conditions, self.states))

        self.grids = Grids(self.cfg.numerical_params.I)

    def _get_state(self, time: float) -> StateFull:
        raise NotImplementedError

    def _get_index(self, time: float) -> int:
        return np.argmin(np.abs(self.times - time))

    @property
    def solid_fraction(self) -> NDArray:
        return _get_array_data("solid_fraction", self.states)

    @property
    def liquid_fraction(self) -> NDArray:
        return _get_array_data("liquid_fraction", self.states)

    @property
    def temperature(self) -> NDArray:
        return _get_array_data("temperature", self.states)

    @property
    def liquid_salinity(self) -> NDArray:
        return _get_array_data("liquid_salinity", self.states)

    @property
    def dissolved_gas(self) -> NDArray:
        return _get_array_data("dissolved_gas", self.states)

    def get_spectral_irradiance(self, time: float) -> oi.SpectralIrradiance:
        if not isinstance(self.cfg.forcing_config, RadForcing):
            raise TypeError("Simulation was not run with radiative forcing")

        return run_two_stream_model(
            self.states_bcs[self._get_index(time)], self.cfg, self.grids
        )

    def total_albedo(self, time: float) -> float:
        """Total albedo including the effect of the surface scattering layer if present,
        if not present then the penetration fraction is 1 and so we regain just albedo
        calculated from the two stream radiative transfer model"""
        spec_irrad = self.get_spectral_irradiance(time)
        ice_albedo = oi.integrate_over_SW(spec_irrad).albedo
        PEN = get_SW_penetration_fraction(
            self.states_bcs[self._get_index(time)], self.cfg
        )
        return 1 - PEN * (1 - ice_albedo)

    def total_transmittance(self, time: float) -> float:
        """Total spectrally integrated transmittance"""
        spec_irrad = self.get_spectral_irradiance(time)
        return oi.integrate_over_SW(spec_irrad).transmittance


@dataclass
class EQMResults(_BaseResults):
    bulk_gas: NDArray

    def _get_state(self, time: float) -> EQMStateFull:
        index = np.argmin(np.abs(self.times - time))
        state = EQMState(
            self.times[index],
            self.enthalpy[:, index],
            self.salt[:, index],
            self.bulk_gas[:, index],
        )
        enthalpy_method = get_enthalpy_method(self.cfg)
        return enthalpy_method(state)

    @property
    def gas_fraction(self) -> NDArray:
        return _get_array_data("gas_fraction", self.states)


@dataclass
class DISEQResults(_BaseResults):
    bulk_dissolved_gas: NDArray
    gas_fraction: NDArray

    def _get_state(self, time: float) -> DISEQStateFull:
        index = np.argmin(np.abs(self.times - time))
        state = DISEQState(
            self.times[index],
            self.enthalpy[:, index],
            self.salt[:, index],
            self.bulk_dissolved_gas[:, index],
            self.gas_fraction[:, index],
        )

        enthalpy_method = get_enthalpy_method(self.cfg)
        return enthalpy_method(state)


Results = EQMResults | DISEQResults


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
