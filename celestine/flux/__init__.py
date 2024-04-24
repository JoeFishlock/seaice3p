"""Module for calculating the fluxes using upwind scheme"""

from .bulk_gas_flux import calculate_gas_flux
from .heat_flux import calculate_heat_flux
from .salt_flux import calculate_salt_flux
from .bulk_dissolved_gas_flux import calculate_bulk_dissolved_gas_flux
from .gas_fraction_flux import calculate_gas_fraction_flux
