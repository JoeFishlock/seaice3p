from functools import partial
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from .RJW14.brine_channel_sink_terms import (
    calculate_heat_sink,
    calculate_salt_sink,
    calculate_gas_sink,
    calculate_bulk_dissolved_gas_sink,
)
from .flux import (
    calculate_heat_flux,
    calculate_salt_flux,
    calculate_gas_flux,
    calculate_bulk_dissolved_gas_flux,
    calculate_gas_fraction_flux,
)
from .forcing import calculate_non_dimensional_shortwave_heating
from .state import StateBCs, DISEQStateBCs, EQMStateBCs
from .velocities.velocities import calculate_velocities


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


def _calculate_brine_convection_sink(state_BCs: StateBCs, cfg, grids) -> NDArray:
    """TODO: check the sink terms for bulk_dissolved_gas and gas fraction

    For now neglect the coupling of bubbles to the horizontal or vertical flow
    """
    heat_sink = calculate_heat_sink(state_BCs, cfg, grids)
    salt_sink = calculate_salt_sink(state_BCs, cfg, grids)
    match state_BCs:
        case EQMStateBCs():
            gas_sink = calculate_gas_sink(state_BCs, cfg, grids)
            return np.hstack((heat_sink, salt_sink, gas_sink))
        case DISEQStateBCs():
            bulk_dissolved_gas_sink = calculate_bulk_dissolved_gas_sink(
                state_BCs, cfg, grids
            )
            gas_fraction_sink = np.zeros_like(heat_sink)
            return np.hstack(
                (heat_sink, salt_sink, bulk_dissolved_gas_sink, gas_fraction_sink)
            )
        case _:
            raise NotImplementedError


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


def _calculate_nucleation(state_BCs: StateBCs, cfg):
    """implement nucleation term"""
    zeros = np.zeros_like(state_BCs.enthalpy[1:-1])
    match state_BCs:
        case EQMStateBCs():
            return np.hstack((zeros, zeros, zeros))
        case DISEQStateBCs():
            chi = cfg.physical_params.expansion_coefficient
            Da = cfg.physical_params.damkohler_number
            centers = np.s_[1:-1]
            bulk_dissolved_gas = state_BCs.bulk_dissolved_gas[centers]
            liquid_fraction = state_BCs.liquid_fraction[centers]
            saturation = chi * liquid_fraction
            gas_fraction = state_BCs.gas_fraction[centers]

            is_saturated = bulk_dissolved_gas > saturation
            nucleation = np.full_like(bulk_dissolved_gas, np.NaN)
            nucleation[is_saturated] = Da * (
                bulk_dissolved_gas[is_saturated] - saturation[is_saturated]
            )
            nucleation[~is_saturated] = -Da * gas_fraction[~is_saturated]

            return np.hstack(
                (
                    zeros,
                    zeros,
                    -nucleation,
                    nucleation,
                )
            )
        case _:
            raise NotImplementedError


def _calculate_dz_fluxes(state_BCs: StateBCs, Wl, Vg, V, cfg, grids):
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


def _calculate_radiative_heating(state_BCs: StateBCs, cfg, grids):
    """Calculate internal shortwave heating source for enthalpy equation.

    Stack with a zero source term for salt, bubble and dissolved gas equation.
    """
    heating = calculate_non_dimensional_shortwave_heating(state_BCs, cfg, grids)
    match state_BCs:
        case EQMStateBCs():
            return np.hstack(
                (
                    heating,
                    np.zeros_like(heating),
                    np.zeros_like(heating),
                )
            )

        case DISEQStateBCs():
            return np.hstack(
                (
                    heating,
                    np.zeros_like(heating),
                    np.zeros_like(heating),
                    np.zeros_like(heating),
                )
            )
        case _:
            raise NotImplementedError


def calculate_equations(state_BCs: StateBCs, cfg, grids):
    Vg, Wl, V = calculate_velocities(state_BCs, cfg)
    Vg = _prevent_gas_rise_into_saturated_cell(Vg, state_BCs)

    dz_fluxes = _calculate_dz_fluxes(state_BCs, Wl, Vg, V, cfg, grids)
    brine_convection_sink = _calculate_brine_convection_sink(state_BCs, cfg, grids)
    nucleation = _calculate_nucleation(state_BCs, cfg)
    radiative_heating = _calculate_radiative_heating(state_BCs, cfg, grids)

    if cfg.forcing_config.SW_internal_heating:
        return -dz_fluxes - brine_convection_sink + nucleation + radiative_heating

    return -dz_fluxes - brine_convection_sink + nucleation
