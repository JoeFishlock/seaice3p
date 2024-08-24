from typing import Callable
import numpy as np
from numpy.typing import NDArray
from .RJW14 import get_brine_convection_sink
from .nucleation import get_nucleation
from .flux import get_dz_fluxes
from .radiative_heating import get_radiative_heating
from ..state import StateBCs
from .velocities import calculate_velocities


def _prevent_gas_rise_into_saturated_cell(Vg, state_BCs: StateBCs) -> NDArray:
    """Modify the gas interstitial velocity to prevent bubble rise into a cell which
    is already theoretically saturated with gas.

    From the state with boundary conditions calculate the gas and solid fraction in the
    cells (except at lower ghost cell). If any of these are such that there is more gas
    fraction than pore space available then set gas insterstitial velocity to zero on
    the edge below. Make sure the very top boundary velocity is not changed as we want
    to always alow flux to the atmosphere regardless of the boundary conditions imposed.

    :param Vg: gas insterstitial velocity on cell edges
    :type Vg: Numpy array (size I+1)
    :param state_BCs: state of system with boundary conditions
    :type state_BCs: celestine.state.StateBCs
    :return: filtered gas interstitial velocities on edges to prevent gas rise into a
        fully gas saturated cell

    """
    gas_fraction_above = state_BCs.gas_fraction[1:]
    solid_fraction_above = 1 - state_BCs.liquid_fraction[1:]
    filtered_Vg = np.where(gas_fraction_above + solid_fraction_above >= 1, 0, Vg)
    filtered_Vg[-1] = Vg[-1]
    return filtered_Vg


def get_equations(cfg, grids) -> Callable[[StateBCs], NDArray]:
    dz_fluxes = get_dz_fluxes(cfg, grids)
    brine_convection_sink = get_brine_convection_sink(cfg, grids)
    nucleation = get_nucleation(cfg)
    radiative_heating = get_radiative_heating(cfg, grids)

    def equations(state_BCs: StateBCs) -> NDArray:
        Vg, Wl, V = calculate_velocities(state_BCs, cfg)
        Vg = _prevent_gas_rise_into_saturated_cell(Vg, state_BCs)

        if cfg.forcing_config.SW_internal_heating:
            return (
                -dz_fluxes(state_BCs, Wl, Vg, V)
                - brine_convection_sink(state_BCs)
                + nucleation(state_BCs)
                + radiative_heating(state_BCs)
            )

        return (
            -dz_fluxes(state_BCs, Wl, Vg, V)
            - brine_convection_sink(state_BCs)
            + nucleation(state_BCs)
        )

    return equations
