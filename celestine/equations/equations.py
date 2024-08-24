import numpy as np
from numpy.typing import NDArray
from .RJW14 import calculate_brine_convection_sink
from .nucleation import calculate_nucleation
from .flux import calculate_dz_fluxes
from .radiative_heating import calculate_radiative_heating
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


# Idea to improve performance by returning the appropriate function for ode solve
# determined ahead of time by the configuration instead of checking at each timestep
# which model we are running
# def _get_brine_convection_sink(cfg, grids) -> Callable[[StateBCs], NDArray]:
#     heat_sink = partial(calculate_heat_sink, cfg=cfg, grids=grids)
#     salt_sink = partial(calculate_salt_sink, cfg=cfg, grids=grids)
#     gas_sink = partial(calculate_gas_sink, cfg=cfg, grids=grids)

#     def _EQM(state_BCs: StateBCs) -> NDArray:
#         return np.hstack(
#             (heat_sink(state_BCs), salt_sink(state_BCs), gas_sink(state_BCs))
#         )

#     if cfg.model == "EQM":
#         return _EQM

#     bulk_dissolved_gas_sink = partial(
#         calculate_bulk_dissolved_gas_sink, cfg=cfg, grids=grids
#     )
#     gas_fraction_sink = lambda S: np.zeros_like(heat_sink(S))

#     def _DISEQ(state_BCs: StateBCs) -> NDArray:
#         return np.hstack(
#             (
#                 heat_sink(state_BCs),
#                 salt_sink(state_BCs),
#                 bulk_dissolved_gas_sink(state_BCs),
#                 gas_fraction_sink(state_BCs),
#             )
#         )

#     if cfg.model == "DISEQ":
#         return _DISEQ
#     raise NotImplementedError


def calculate_equations(state_BCs: StateBCs, cfg, grids):
    Vg, Wl, V = calculate_velocities(state_BCs, cfg)
    Vg = _prevent_gas_rise_into_saturated_cell(Vg, state_BCs)

    dz_fluxes = calculate_dz_fluxes(state_BCs, Wl, Vg, V, cfg, grids)
    brine_convection_sink = calculate_brine_convection_sink(state_BCs, cfg, grids)
    nucleation = calculate_nucleation(state_BCs, cfg)

    if cfg.forcing_config.SW_internal_heating:
        radiative_heating = calculate_radiative_heating(state_BCs, cfg, grids)
        return -dz_fluxes - brine_convection_sink + nucleation + radiative_heating

    return -dz_fluxes - brine_convection_sink + nucleation
