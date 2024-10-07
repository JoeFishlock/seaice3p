"""Module for calculating the fluxes using upwind scheme"""
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .bulk_gas_flux import calculate_gas_flux
from .heat_flux import calculate_heat_flux
from .salt_flux import calculate_salt_flux
from .bulk_dissolved_gas_flux import calculate_bulk_dissolved_gas_flux
from .gas_fraction_flux import calculate_gas_fraction_flux

from ...state import StateBCs, EQMStateBCs, DISEQStateBCs
from ...params import Config, EQMPhysicalParams, DISEQPhysicalParams
from ...grids import Grids


def get_dz_fluxes(
    cfg: Config, grids: Grids
) -> Callable[[StateBCs, NDArray, NDArray, NDArray], NDArray]:
    fun_map = {
        EQMPhysicalParams: _EQM_dz_fluxes,
        DISEQPhysicalParams: _DISEQ_dz_fluxes,
    }

    def dz_fluxes(state_BCs: StateBCs, Wl, Vg, V) -> NDArray:
        return fun_map[type(cfg.physical_params)](state_BCs, Wl, Vg, V, cfg, grids)

    return dz_fluxes


def _EQM_dz_fluxes(state_BCs: EQMStateBCs, Wl, Vg, V, cfg, grids) -> NDArray:
    D_g, D_e = grids.D_g, grids.D_e
    dz = lambda flux: np.matmul(D_e, flux)
    heat_flux = calculate_heat_flux(state_BCs, Wl, V, D_g, cfg)
    salt_flux = calculate_salt_flux(state_BCs, Wl, V, D_g, cfg)
    gas_flux = calculate_gas_flux(state_BCs, Wl, V, Vg, D_g, cfg)
    return np.hstack((dz(heat_flux), dz(salt_flux), dz(gas_flux)))


def _DISEQ_dz_fluxes(state_BCs: DISEQStateBCs, Wl, Vg, V, cfg, grids) -> NDArray:
    D_g, D_e = grids.D_g, grids.D_e
    dz = lambda flux: np.matmul(D_e, flux)
    heat_flux = calculate_heat_flux(state_BCs, Wl, V, D_g, cfg)
    salt_flux = calculate_salt_flux(state_BCs, Wl, V, D_g, cfg)
    bulk_dissolved_gas_flux = calculate_bulk_dissolved_gas_flux(
        state_BCs, Wl, V, D_g, cfg
    )
    gas_fraction_flux = calculate_gas_fraction_flux(state_BCs, V, Vg, D_g, cfg)
    return np.hstack(
        (
            dz(heat_flux),
            dz(salt_flux),
            dz(bulk_dissolved_gas_flux),
            dz(gas_fraction_flux),
        )
    )
