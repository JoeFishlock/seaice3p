"""Calculate internal shortwave radiative heating due to oil droplets"""

from typing import Callable
import numpy as np
from numpy.typing import NDArray
import oilrad as oi
from ..grids import Grids, average
from ..params import (
    Config,
    EQMPhysicalParams,
    DISEQPhysicalParams,
    RadForcing,
    ERA5Forcing,
)
from ..params.dimensional import (
    DimensionalBackgroundOilHeating,
    DimensionalMobileOilHeating,
    DimensionalNoHeating,
)
from ..forcing import get_SW_forcing, get_SW_penetration_fraction
from ..state import StateBCs, EQMStateBCs, DISEQStateBCs
from ..oil_mass import convert_gas_fraction_to_oil_mass_ratio


def get_radiative_heating(cfg: Config, grids: Grids) -> Callable[[StateBCs], NDArray]:
    """Calculate internal shortwave heating source for enthalpy equation.

    if the RadForcing object is given as the forcing config then calculates internal
    heating based on the object given in the configuration for oil_heating.

    If another forcing is chosen then just returns a function to create an array of
    zeros as no internal heating is calculated.
    """
    fun_map = {
        EQMPhysicalParams: _EQM_radiative_heating,
        DISEQPhysicalParams: _DISEQ_radiative_heating,
    }

    def radiative_heating(state_BCs: StateBCs) -> NDArray:
        return fun_map[type(cfg.physical_params)](state_BCs, cfg, grids)

    return radiative_heating


def _EQM_radiative_heating(
    state_BCs: EQMStateBCs, cfg: Config, grids: Grids
) -> NDArray:
    heating = _calculate_non_dimensional_shortwave_heating(state_BCs, cfg, grids)
    return np.hstack(
        (
            heating,
            np.zeros_like(heating),
            np.zeros_like(heating),
        )
    )


def _DISEQ_radiative_heating(
    state_BCs: DISEQStateBCs, cfg: Config, grids: Grids
) -> NDArray:
    heating = _calculate_non_dimensional_shortwave_heating(state_BCs, cfg, grids)
    return np.hstack(
        (
            heating,
            np.zeros_like(heating),
            np.zeros_like(heating),
            np.zeros_like(heating),
        )
    )


def run_two_stream_model(
    state_bcs: StateBCs, cfg: Config, grids: Grids
) -> oi.SpectralIrradiance:

    SW_RANGE = (
        cfg.forcing_config.SW_forcing.SW_min_wavelength,
        cfg.forcing_config.SW_forcing.SW_max_wavelength,
    )
    NUM_WAVELENGTH_SAMPLES = cfg.forcing_config.SW_forcing.num_wavelength_samples
    wavelengths = np.geomspace(SW_RANGE[0], SW_RANGE[1], NUM_WAVELENGTH_SAMPLES)

    match cfg.forcing_config.oil_heating:
        case DimensionalBackgroundOilHeating():
            oil_mass_ratio = np.full_like(
                grids.edges, cfg.forcing_config.oil_heating.oil_mass_ratio
            )
            MEDIAN_DROPLET_RADIUS_MICRONS = (
                cfg.forcing_config.oil_heating.median_oil_droplet_radius
            )

        case DimensionalMobileOilHeating():
            oil_mass_ratio = convert_gas_fraction_to_oil_mass_ratio(
                average(state_bcs.gas_fraction),
                cfg.scales.gas_density,
                cfg.scales.ice_density,
            )
            MEDIAN_DROPLET_RADIUS_MICRONS = (
                cfg.scales.pore_radius * cfg.bubble_params.bubble_radius_scaled * 1e6
            )
        case _:
            raise NotImplementedError()

    model = oi.InfiniteLayerModel(
        grids.edges * cfg.scales.lengthscale,
        wavelengths,
        oil_mass_ratio=oil_mass_ratio,
        ice_type=cfg.forcing_config.oil_heating.ice_type,
        median_droplet_radius_in_microns=MEDIAN_DROPLET_RADIUS_MICRONS,
        liquid_fraction=average(state_bcs.liquid_fraction),
        fast_solve=cfg.forcing_config.oil_heating.fast_solve,
        wavelength_cutoff=cfg.forcing_config.oil_heating.wavelength_cutoff,
    )
    return oi.solve_two_stream_model(model)


def _calculate_non_dimensional_shortwave_heating(
    state_bcs: StateBCs, cfg: Config, grids: Grids
) -> NDArray:
    """Calculate internal shortwave heating due to oil droplets on center grid

    Assumes a configuration with the RadForcing object as the forcing config is
    passed."""
    # If we don't have radiative forcing then just return array of zeros for heating
    if not (
        isinstance(cfg.forcing_config, RadForcing)
        or isinstance(cfg.forcing_config, ERA5Forcing)
    ):
        return np.zeros_like(grids.centers)

    if isinstance(cfg.forcing_config.oil_heating, DimensionalNoHeating):
        return np.zeros_like(grids.centers)

    dimensionless_incident_SW = cfg.scales.convert_from_dimensional_heat_flux(
        get_SW_penetration_fraction(state_bcs, cfg)
        * get_SW_forcing(state_bcs.time, cfg)
    )
    # If no incident SW don't need to run radiation model
    if dimensionless_incident_SW == 0:
        return np.zeros_like(grids.centers)

    spectral_irradiances = run_two_stream_model(state_bcs, cfg, grids)
    spectrum = oi.BlackBodySpectrum(
        cfg.forcing_config.SW_forcing.SW_min_wavelength,
        cfg.forcing_config.SW_forcing.SW_max_wavelength,
    )
    integrated_irradiance = oi.integrate_over_SW(spectral_irradiances, spectrum)

    dz_dF_net = grids.D_e @ integrated_irradiance.net_irradiance
    return dimensionless_incident_SW * dz_dF_net
