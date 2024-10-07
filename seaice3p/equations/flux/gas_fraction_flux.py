"""Calculate gas phase fluxes for disequilibrium model"""

from .bulk_gas_flux import (
    calculate_bubble_gas_flux,
    calculate_diffusive_gas_bubble_flux,
    calculate_frame_advection_gas_flux,
)
from ...params import Config


def calculate_gas_fraction_flux(state_BCs, V, Vg, D_g, cfg: Config):
    gas_fraction = state_BCs.gas_fraction
    liquid_fraction = state_BCs.liquid_fraction
    gas_fraction_flux = (
        calculate_bubble_gas_flux(gas_fraction, Vg)
        + calculate_frame_advection_gas_flux(gas_fraction, V)
        + calculate_diffusive_gas_bubble_flux(gas_fraction, liquid_fraction, D_g, cfg)
    )
    return gas_fraction_flux
