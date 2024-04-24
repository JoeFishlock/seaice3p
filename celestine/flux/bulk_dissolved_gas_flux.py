"""calculate the flux terms for the dissolved gas equation in DISEQ model"""
from .bulk_gas_flux import (
    calculate_diffusive_gas_flux,
    calculate_advective_dissolved_gas_flux,
    calculate_frame_advection_gas_flux,
)


def calculate_bulk_dissolved_gas_flux(state_BCs, Wl, V, D_g, cfg):
    dissolved_gas = state_BCs.dissolved_gas
    liquid_fraction = state_BCs.liquid_fraction
    bulk_dissolved_gas = state_BCs.bulk_dissolved_gas

    bulk_dissolved_gas_flux = (
        calculate_diffusive_gas_flux(dissolved_gas, liquid_fraction, D_g, cfg)
        + calculate_advective_dissolved_gas_flux(dissolved_gas, Wl, cfg)
        + calculate_frame_advection_gas_flux(bulk_dissolved_gas, V)
    )
    return bulk_dissolved_gas_flux
