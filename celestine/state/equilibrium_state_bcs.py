from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from ..forcing import calculate_non_dimensional_shortwave_heating
from ..flux import calculate_gas_flux, calculate_heat_flux, calculate_salt_flux
from ..RJW14 import (
    calculate_heat_sink,
    calculate_salt_sink,
    calculate_gas_sink,
)
from ..velocities import calculate_velocities


def prevent_gas_rise_into_saturated_cell(Vg, state_BCs):
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


@dataclass(frozen=True)
class EQMStateBCs:
    """Stores information needed for solution at one timestep with BCs on ghost
    cells as well

    Initialiase the prime variables for the solver:
    enthalpy, bulk salinity and bulk air
    """

    time: float
    enthalpy: NDArray
    salt: NDArray
    gas: NDArray

    temperature: NDArray
    liquid_salinity: NDArray
    dissolved_gas: NDArray
    gas_fraction: NDArray
    liquid_fraction: NDArray

    def _calculate_brine_convection_sink(self, cfg, grids):
        heat_sink = calculate_heat_sink(self, cfg, grids)
        salt_sink = calculate_salt_sink(self, cfg, grids)
        gas_sink = calculate_gas_sink(self, cfg, grids)
        return np.hstack((heat_sink, salt_sink, gas_sink))

    def _calculate_dz_fluxes(self, Wl, Vg, V, cfg, grids):
        D_g = grids.D_g
        D_e = grids.D_e
        heat_flux = calculate_heat_flux(self, Wl, V, D_g, cfg)
        salt_flux = calculate_salt_flux(self, Wl, V, D_g, cfg)
        gas_flux = calculate_gas_flux(self, Wl, V, Vg, D_g, cfg)
        dz = lambda flux: np.matmul(D_e, flux)
        return np.hstack((dz(heat_flux), dz(salt_flux), dz(gas_flux)))

    def _calculate_radiative_heating(self, grids):
        """Calculate internal shortwave heating source for enthalpy equation.

        Stack with a zero source term for salt and bulk gas equation.
        """
        heating = calculate_non_dimensional_shortwave_heating(self, grids)
        return np.hstack(
            (
                heating,
                np.zeros_like(heating),
                np.zeros_like(heating),
            )
        )

    def calculate_equation(self, cfg, grids):
        D_e, D_g = grids.D_e, grids.D_g
        Vg, Wl, V = calculate_velocities(self, cfg)
        Vg = prevent_gas_rise_into_saturated_cell(Vg, self)

        if cfg.forcing_config.SW_internal_heating:
            return (
                -self._calculate_dz_fluxes(Wl, Vg, V, cfg, grids)
                - self._calculate_brine_convection_sink(cfg, grids)
                + self._calculate_radiative_heating(grids)
            )

        return -self._calculate_dz_fluxes(
            Wl, Vg, V, cfg, grids
        ) - self._calculate_brine_convection_sink(cfg, grids)
