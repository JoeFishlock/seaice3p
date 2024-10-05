from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import numpy as np
from numpy.typing import NDArray
import oilrad as oi
from scipy.integrate import trapezoid


from . import (
    Config,
    DimensionalParams,
    get_config,
    DISEQPhysicalParams,
    EQMPhysicalParams,
    Grids,
    RadForcing,
    ERA5Forcing,
)
from .state import EQMState, DISEQState, DISEQStateFull, EQMStateFull, StateFull
from .enthalpy_method import get_enthalpy_method
from .forcing.boundary_conditions import get_boundary_conditions
from .forcing import get_SW_penetration_fraction, get_SW_forcing
from .equations.radiative_heating import run_two_stream_model
from .oil_mass import convert_gas_fraction_to_oil_mass_ratio
from .forcing.surface_energy_balance.turbulent_heat_flux import (
    calculate_sensible_heat_flux,
    calculate_latent_heat_flux,
)
from .forcing.surface_energy_balance.surface_energy_balance import (
    _convert_non_dim_temperature_to_kelvin,
    _calculate_emissivity,
    STEFAN_BOLTZMANN,
    find_ghost_cell_temperature,
)
from .forcing.radiative_forcing import get_LW_forcing


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

    def _is_ice(self, time: float) -> NDArray:
        """Boolean mask True where ice is present on center grid cells at given
        non-dimensional time"""
        return self.liquid_fraction[:, self._get_index(time)] < 1

    def _top_cell_is_ice(self, time: float) -> bool:
        """Return True if top cell is ice or False if liquid"""
        return self._is_ice(time)[-1]

    @property
    def dates(self) -> List[datetime]:
        if hasattr(self.cfg.forcing_config, "start_date"):
            days = self.times * self.cfg.scales.time_scale
            start_date = datetime.strptime(
                self.cfg.forcing_config.start_date, "%Y-%m-%d"
            )
            return [timedelta(days=day) + start_date for day in days]
        else:
            raise AttributeError("forcing has no start date")

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

    @property
    def dimensional_meltpond_onset_time(self) -> float:
        """Get meltpond onset time from start of simulation in days"""
        top_liquid_fraction = self.liquid_fraction[-1, :]
        times_with_meltpond = self.times[top_liquid_fraction == 1]
        if times_with_meltpond.size == 0:
            return np.NaN
        return self.cfg.scales.convert_to_dimensional_time(times_with_meltpond[0])

    @property
    def oil_mass_ratio(self) -> NDArray:
        """in ng/g"""
        return convert_gas_fraction_to_oil_mass_ratio(
            self.gas_fraction, self.cfg.scales.gas_density, self.cfg.scales.ice_density
        )

    @property
    def bulk_argon(self) -> NDArray:
        """in mircomole Ar/L"""
        scales = self.cfg.scales
        return scales.convert_dimensional_bulk_air_to_argon_content(
            scales.convert_to_dimensional_bulk_gas(self.bulk_gas)
        )

    def get_spectral_irradiance(self, time: float) -> oi.SpectralIrradiance:
        if not (
            isinstance(self.cfg.forcing_config, RadForcing)
            or isinstance(self.cfg.forcing_config, ERA5Forcing)
        ):
            raise TypeError("Simulation was not run with radiative forcing")

        return run_two_stream_model(
            self.states_bcs[self._get_index(time)], self.cfg, self.grids
        )

    def total_albedo(self, time: float) -> float:
        """Total albedo including the effect of the surface scattering layer if present,
        if not present then the penetration fraction is 1 and so we regain just albedo
        calculated from the two stream radiative transfer model"""
        spec_irrad = self.get_spectral_irradiance(time)
        spectrum = oi.BlackBodySpectrum(
            self.cfg.forcing_config.SW_forcing.SW_min_wavelength,
            self.cfg.forcing_config.SW_forcing.SW_max_wavelength,
        )
        ice_albedo = oi.integrate_over_SW(spec_irrad, spectrum).albedo
        PEN = get_SW_penetration_fraction(
            self.states_bcs[self._get_index(time)], self.cfg
        )
        return 1 - PEN * (1 - ice_albedo)

    def total_transmittance(self, time: float) -> float:
        """Total spectrally integrated transmittance"""
        spec_irrad = self.get_spectral_irradiance(time)
        spectrum = oi.BlackBodySpectrum(
            self.cfg.forcing_config.SW_forcing.SW_min_wavelength,
            self.cfg.forcing_config.SW_forcing.SW_max_wavelength,
        )
        return oi.integrate_over_SW(spec_irrad, spectrum).transmittance

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

        # if no ice then no boundary
        if np.all(liquid_fraction == 1):
            return np.NaN

        is_ice_centers = self._is_ice(time)
        is_ice_edges = np.hstack((is_ice_centers, is_ice_centers[-1]))
        return self.grids.edges[is_ice_edges][0]

    def ice_meltpond_boundary(self, time: float) -> float:
        index = self._get_index(time)
        liquid_fraction = self.liquid_fraction[:, index]

        # if no ice then no meltpond
        if np.all(liquid_fraction == 1):
            return np.NaN

        is_ice_centers = self._is_ice(time)
        is_ice_edges = np.hstack((is_ice_centers[0], is_ice_centers))
        return self.grids.edges[is_ice_edges][-1]

    def ice_thickness(self, time: float) -> float:
        index = self._get_index(time)
        liquid_fraction = self.liquid_fraction[:, index]

        # if no ice no thickness
        if np.all(liquid_fraction == 1):
            return 0
        return self.ice_meltpond_boundary(time) - self.ice_ocean_boundary(time)

    def integrated_solid_fraction(self, time: float) -> float:
        return _integrate(
            self.solid_fraction[:, self._get_index(time)],
            self.cfg.numerical_params.step,
        )

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

    @property
    def dimensional_salinity_dependent_liquid_density(self) -> NDArray:
        reference_liquid_density = self.cfg.scales.liquid_density
        return reference_liquid_density * (
            1
            + self.cfg.scales.haline_contraction_coefficient
            * self.cfg.scales.salinity_difference
            * self.liquid_salinity
        )

    @property
    def dimensional_bulk_density(self) -> NDArray:
        return (
            (self.corrected_solid_fraction * self.cfg.scales.ice_density)
            + (
                self.corrected_liquid_fraction
                * self.dimensional_salinity_dependent_liquid_density
            )
            + (self.gas_fraction * self.cfg.scales.gas_density)
        )

    def dimensional_ice_average_bulk_density(self, time: float) -> float:
        index = self._get_index(time)
        is_ice = self._is_ice(time)
        bulk_density = self.dimensional_bulk_density[is_ice, index]
        if bulk_density.size == 0:
            return np.NaN

        return np.mean(bulk_density)

    def total_bulk_gas_content(self, time: float) -> float:
        """To get dimensional bulk gas in domain multiply by
        gas_density * lengthscale
        """
        index = self._get_index(time)
        return _integrate(self.bulk_gas[:, index], self.cfg.numerical_params.step)

    def ice_bulk_gas_content(self, time: float) -> float:
        """To get dimensional bulk gas in ice multiply by
        gas_density * lengthscale
        """
        index = self._get_index(time)
        is_ice = self._is_ice(time)
        return _integrate(self.bulk_gas[is_ice, index], self.cfg.numerical_params.step)

    def surface_temp_K(self, time: float) -> float:
        """Return surface temperature in K"""
        index = self._get_index(time)
        ghost_cell_temp = find_ghost_cell_temperature(self.states[index], self.cfg)
        return _convert_non_dim_temperature_to_kelvin(
            self.cfg, 0.5 * (ghost_cell_temp + self.temperature[-1, index])
        )

    def sensible_heat_flux(self, time: float) -> float:
        """W/m2"""
        return calculate_sensible_heat_flux(
            self.cfg, time, self._top_cell_is_ice(time), self.surface_temp_K(time)
        )

    def latent_heat_flux(self, time: float) -> float:
        """W/m2"""
        return calculate_latent_heat_flux(
            self.cfg, time, self._top_cell_is_ice(time), self.surface_temp_K(time)
        )

    def emitted_LW(self, time: float) -> float:
        """W/m2 radiated away from ice surface"""
        emissivity = _calculate_emissivity(self.cfg, self._top_cell_is_ice(time))
        return emissivity * STEFAN_BOLTZMANN * self.surface_temp_K(time) ** 4

    def net_LW(self, time: float) -> float:
        """W/m2 net into ice"""
        incident_LW = get_LW_forcing(time, self.cfg)
        return incident_LW - self.emitted_LW(time)

    def surface_heat_flux(self, time: float) -> float:
        """W/m2 net into ice"""
        return (
            self.net_LW(time)
            + self.sensible_heat_flux(time)
            + self.latent_heat_flux(time)
        )


def _integrate(quantity: NDArray, step: float) -> float:
    if quantity.size == 0:
        return np.NaN
    return step * np.sum(quantity)


@dataclass
class EQMResults(_BaseResults):
    bulk_gas: NDArray

    def _get_state(self, time: float) -> EQMStateFull:
        index = self._get_index(time)
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
        index = self._get_index(time)
        state = DISEQState(
            self.times[index],
            self.enthalpy[:, index],
            self.salt[:, index],
            self.bulk_dissolved_gas[:, index],
            self.gas_fraction[:, index],
        )

        enthalpy_method = get_enthalpy_method(self.cfg)
        return enthalpy_method(state)

    @property
    def bulk_gas(self) -> NDArray:
        """Dimensionless bulk gas the same as the EQM model"""
        return self.bulk_dissolved_gas + self.gas_fraction


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
