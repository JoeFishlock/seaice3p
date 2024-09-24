from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import oilrad as oi
from oilrad.black_body import get_normalised_black_body_spectrum
from scipy.integrate import trapezoid


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
from .forcing import get_SW_penetration_fraction, get_SW_forcing
from .equations.radiative_heating import run_two_stream_model
from .grids import calculate_ice_ocean_boundary_depth


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

    def dimensional_PAR_transmittance(self, time: float) -> float:
        """Total photosynthetically active radiation transmitted through ice in W/m2"""
        spec_irrad = self.get_spectral_irradiance(time)
        is_PAR = (spec_irrad.wavelengths >= 400) & (spec_irrad.wavelengths <= 700)
        PAR_wavelengths = spec_irrad.wavelengths[is_PAR]
        if PAR_wavelengths.size == 0:
            raise RuntimeError("Not enough points in wavelength space to integrate")

        PAR_irrad = oi.SpectralIrradiance(
            spec_irrad.z,
            PAR_wavelengths,
            spec_irrad.upwelling[:, is_PAR],
            spec_irrad.downwelling[:, is_PAR],
            ice_base_index=spec_irrad.ice_base_index,
        )
        SW = get_SW_forcing(time, self.cfg)
        spectrum = get_normalised_black_body_spectrum(
            (spec_irrad.wavelengths[0], spec_irrad.wavelengths[-1])
        )

        if PAR_wavelengths.size == 1:
            return (
                SW
                * spectrum(PAR_irrad.wavelengths[0])
                * PAR_irrad.transmittance[0]
                * (700 - 400)
            )

        return SW * trapezoid(
            PAR_irrad.transmittance * spectrum(PAR_irrad.wavelengths), PAR_wavelengths
        )

    def ice_ocean_boundary(self, time: float) -> float:
        index = self._get_index(time)
        liquid_fraction = self.liquid_fraction[:, index]
        return -calculate_ice_ocean_boundary_depth(liquid_fraction, self.grids.edges)

    def ice_meltpond_boundary(self, time: float) -> float:
        index = self._get_index(time)
        liquid_fraction = self.liquid_fraction[:, index]

        # if no ice then no meltpond
        if np.all(liquid_fraction == 1):
            return np.NaN

        is_ice = np.where(liquid_fraction < 1)[0]
        top_index = is_ice[-1]
        boundary = self.grids.edges[top_index + 1]

        # if no meltpond is present we are just detecting ice_ocean_boundary
        if boundary == self.ice_ocean_boundary(time):
            return 0

        return boundary

    def ice_thickness(self, time: float) -> float:
        index = self._get_index(time)
        liquid_fraction = self.liquid_fraction[:, index]

        # if no ice no thickness
        if np.all(liquid_fraction == 1):
            return 0
        return self.ice_meltpond_boundary(time) - self.ice_ocean_boundary(time)

    def integrated_solid_fraction(self, time: float) -> float:
        index = self._get_index(time)
        return trapezoid(self.solid_fraction[:, index], self.grids.centers)

    @property
    def corrected_solid_fraction(self) -> NDArray:
        """Adjusted so that corrected_solid_fraction + corrected_liquid_fraction + gas_fraction = 1"""
        corrected_solid_fraction = np.empty_like(self.solid_fraction)
        is_frozen = self.liquid_fraction == 0

        corrected_solid_fraction[is_frozen] = (
            self.solid_fraction[is_frozen] - self.gas_fraction[is_frozen]
        )
        corrected_solid_fraction[~is_frozen] = self.solid_fraction[~is_frozen]
        if np.any(corrected_solid_fraction < 0):
            raise ValueError("Corrected solid fraction is negative")
        return corrected_solid_fraction

    @property
    def corrected_liquid_fraction(self) -> NDArray:
        """Adjusted so that corrected_solid_fraction + corrected_liquid_fraction + gas_fraction = 1"""
        corrected_liquid_fraction = np.empty_like(self.liquid_fraction)
        is_frozen = self.liquid_fraction == 0

        corrected_liquid_fraction[is_frozen] = self.liquid_fraction[is_frozen]
        corrected_liquid_fraction[~is_frozen] = (
            self.liquid_fraction[~is_frozen] - self.gas_fraction[~is_frozen]
        )
        if np.any(corrected_liquid_fraction < 0):
            raise ValueError("Corrected liquid fraction is negative")
        return corrected_liquid_fraction


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
