"""Calculate gas phase fluxes for disequilibrium model"""

from .bulk_gas_flux import calculate_bubble_gas_flux, calculate_frame_advection_gas_flux


def calculate_gas_fraction_flux(state_BCs, V, Vg):
    gas_fraction = state_BCs.gas_fraction
    gas_fraction_flux = calculate_bubble_gas_flux(
        gas_fraction, Vg
    ) + calculate_frame_advection_gas_flux(gas_fraction, V)
    return gas_fraction_flux
