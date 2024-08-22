from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from ..forcing import calculate_non_dimensional_shortwave_heating
from ..flux import (
    calculate_heat_flux,
    calculate_salt_flux,
    calculate_bulk_dissolved_gas_flux,
    calculate_gas_fraction_flux,
)
from ..RJW14 import (
    calculate_heat_sink,
    calculate_salt_sink,
    calculate_bulk_dissolved_gas_sink,
)
from .equilibrium_state_bcs import prevent_gas_rise_into_saturated_cell
from ..velocities import calculate_velocities


@dataclass(frozen=True)
class DISEQStateBCs:
    """Stores information needed for solution at one timestep with BCs on ghost
    cells as well

    Initialiase the prime variables for the solver:
    enthalpy, bulk salinity and bulk air
    """

    time: float
    enthalpy: NDArray
    salt: NDArray

    temperature: NDArray
    liquid_salinity: NDArray
    dissolved_gas: NDArray
    liquid_fraction: NDArray
    bulk_dissolved_gas: NDArray
    gas_fraction: NDArray

    def _calculate_brine_convection_sink(self, cfg, grids):
        """TODO: check the sink terms for bulk_dissolved_gas and gas fraction

        For now neglect the coupling of bubbles to the horizontal or vertical flow
        """
        heat_sink = calculate_heat_sink(self, cfg, grids)
        salt_sink = calculate_salt_sink(self, cfg, grids)
        bulk_dissolved_gas_sink = calculate_bulk_dissolved_gas_sink(self, cfg, grids)
        gas_fraction_sink = np.zeros_like(heat_sink)
        return np.hstack(
            (heat_sink, salt_sink, bulk_dissolved_gas_sink, gas_fraction_sink)
        )

    def _calculate_nucleation(self, cfg):
        """implement nucleation term"""
        chi = cfg.physical_params.expansion_coefficient
        Da = cfg.physical_params.damkohler_number
        centers = np.s_[1:-1]
        bulk_dissolved_gas = self.bulk_dissolved_gas[centers]
        liquid_fraction = self.liquid_fraction[centers]
        saturation = chi * liquid_fraction
        gas_fraction = self.gas_fraction[centers]

        is_saturated = bulk_dissolved_gas > saturation
        nucleation = np.full_like(bulk_dissolved_gas, np.NaN)
        nucleation[is_saturated] = Da * (
            bulk_dissolved_gas[is_saturated] - saturation[is_saturated]
        )
        nucleation[~is_saturated] = -Da * gas_fraction[~is_saturated]

        return np.hstack(
            (
                np.zeros_like(nucleation),
                np.zeros_like(nucleation),
                -nucleation,
                nucleation,
            )
        )

    def _calculate_dz_fluxes(self, Wl, Vg, V, cfg, grids):
        D_g, D_e = grids.D_g, grids.D_e
        heat_flux = calculate_heat_flux(self, Wl, V, D_g, cfg)
        salt_flux = calculate_salt_flux(self, Wl, V, D_g, cfg)
        bulk_dissolved_gas_flux = calculate_bulk_dissolved_gas_flux(
            self, Wl, V, D_g, cfg
        )
        gas_fraction_flux = calculate_gas_fraction_flux(self, V, Vg)
        dz = lambda flux: np.matmul(D_e, flux)
        return np.hstack(
            (
                dz(heat_flux),
                dz(salt_flux),
                dz(bulk_dissolved_gas_flux),
                dz(gas_fraction_flux),
            )
        )

    def _calculate_radiative_heating(self, grids):
        """Calculate internal shortwave heating source for enthalpy equation.

        Stack with a zero source term for salt, bubble and dissolved gas equation.
        """
        heating = calculate_non_dimensional_shortwave_heating(self, grids)
        return np.hstack(
            (
                heating,
                np.zeros_like(heating),
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
                + self._calculate_nucleation(cfg)
                + self._calculate_radiative_heating(grids)
            )

        return (
            -self._calculate_dz_fluxes(Wl, Vg, V, cfg, grids)
            - self._calculate_brine_convection_sink(cfg, grids)
            + self._calculate_nucleation(cfg)
        )
