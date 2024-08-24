"""Module for calculating the fluxes using upwind scheme"""
import numpy as np
from numpy.typing import NDArray

from .bulk_gas_flux import calculate_gas_flux
from .heat_flux import calculate_heat_flux
from .salt_flux import calculate_salt_flux
from .bulk_dissolved_gas_flux import calculate_bulk_dissolved_gas_flux
from .gas_fraction_flux import calculate_gas_fraction_flux
from ...state import StateBCs, EQMStateBCs, DISEQStateBCs


def calculate_dz_fluxes(state_BCs: StateBCs, Wl, Vg, V, cfg, grids) -> NDArray:
    D_g, D_e = grids.D_g, grids.D_e
    dz = lambda flux: np.matmul(D_e, flux)
    heat_flux = calculate_heat_flux(state_BCs, Wl, V, D_g, cfg)
    salt_flux = calculate_salt_flux(state_BCs, Wl, V, D_g, cfg)
    match state_BCs:
        case EQMStateBCs():
            gas_flux = calculate_gas_flux(state_BCs, Wl, V, Vg, D_g, cfg)
            return np.hstack((dz(heat_flux), dz(salt_flux), dz(gas_flux)))
        case DISEQStateBCs():
            bulk_dissolved_gas_flux = calculate_bulk_dissolved_gas_flux(
                state_BCs, Wl, V, D_g, cfg
            )
            gas_fraction_flux = calculate_gas_fraction_flux(state_BCs, V, Vg)
            return np.hstack(
                (
                    dz(heat_flux),
                    dz(salt_flux),
                    dz(bulk_dissolved_gas_flux),
                    dz(gas_fraction_flux),
                )
            )
        case _:
            raise NotImplementedError
