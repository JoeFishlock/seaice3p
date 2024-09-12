"""Module to compute the turbulent atmospheric sensible and latent heat fluxes

All temperatures are in Kelvin in this module

Refs:
[1] P. D. Taylor and D. L. Feltham, ‘A model of melt pond evolution on sea ice’,
J. Geophys. Res., vol. 109, no. C12, p. 2004JC002361, Dec. 2004,
doi: 10.1029/2004JC002361.

[2] E. E. Ebert and J. A. Curry,
‘An intermediate one-dimensional thermodynamic sea ice model for investigating
ice-atmosphere interactions’,
Journal of Geophysical Research: Oceans, vol. 98, no. C6, pp. 10085–10109, 1993,
doi: 10.1029/93JC00656.
"""
import numpy as np
from ...params import Config

REF_HEIGHT = 10  # m
REF_AIR_DENSITY = 1.275  # kg/m3
REF_AIR_HEAT_CAPACITY = 1005  # J/kg K
GRAVITY = 9.81  # m/s2
LATENT_HEAT_OF_VAPORISATION = 2.501e6  # J/kg

REF_WINDSPEED = 5  # m/s
REF_AIR_TEMP = 273  # K
REF_SPECIFIC_HUMIDITY = 3.6e-3  # kg water / kg air
REF_ATM_PRESSURE = 101.325  # KPa


def calculate_ref_windspeed(cfg: Config, time: float) -> float:
    """return windspeed at reference level above the ice"""
    return REF_WINDSPEED


def calculate_ref_air_temp(cfg: Config, time: float) -> float:
    """return air temperature at reference level above the ice"""
    return REF_AIR_TEMP


def calculate_ref_specific_humidity(cfg: Config, time: float) -> float:
    """return specific humidity at reference level above the ice"""
    return REF_SPECIFIC_HUMIDITY


def calculate_ref_atmospheric_pressure(cfg: Config, time: float) -> float:
    """return atmospheric pressure at reference level above the ice"""
    return REF_ATM_PRESSURE


def calculate_bulk_transfer_coefficient(
    cfg: Config, top_cell_is_ice: bool, time: float, surface_temp: float
) -> float:
    """Calculation of bulk transfer coeff from [2]"""
    if top_cell_is_ice:
        CT0 = 1.3e-3
    else:
        CT0 = 1.0e-3
    BPRIME = 20
    C = 1961 * BPRIME * CT0
    ref_air_temp = calculate_ref_air_temp(cfg, time)
    ref_windspeed = calculate_ref_windspeed(cfg, time)
    Richardson = (
        GRAVITY
        * (ref_air_temp - surface_temp)
        * REF_HEIGHT
        / (ref_air_temp * ref_windspeed**2)
    )
    if Richardson < 0:
        frac = 2 * BPRIME * Richardson / (1 + C * np.sqrt(np.abs(Richardson)))
        return CT0 * (1 - frac)

    return CT0 * (1 + BPRIME * Richardson) ** (-2)


def calculate_surface_specific_humidity(
    cfg: Config, time: float, surface_temp: float
) -> float:
    """Following expression given in [1]"""
    water_vapor_partial_pressure = 2.53e8 * np.exp(-(5420 / surface_temp))
    atm_pressure = calculate_ref_atmospheric_pressure(cfg, time)
    return (
        0.622
        * water_vapor_partial_pressure
        / (atm_pressure - 0.378 * water_vapor_partial_pressure)
    )


def calculate_sensible_heat_flux(
    cfg: Config, time: float, top_cell_is_ice: bool, surface_temp: float
) -> float:
    """Calculate sensible heat flux from [2]"""
    ref_air_temp = calculate_ref_air_temp(cfg, time)
    windspeed = calculate_ref_windspeed(cfg, time)
    bulk_transfer_coeff = calculate_bulk_transfer_coefficient(
        cfg, top_cell_is_ice, time, surface_temp
    )
    return (
        REF_AIR_DENSITY
        * REF_AIR_HEAT_CAPACITY
        * bulk_transfer_coeff
        * windspeed
        * (ref_air_temp - surface_temp)
    )


def calculate_latent_heat_flux(
    cfg: Config, time: float, top_cell_is_ice: bool, surface_temp: float
) -> float:
    """Calculate latent heat flux from [2]"""
    windspeed = calculate_ref_windspeed(cfg, time)
    ref_specific_humidity = calculate_ref_specific_humidity(cfg, time)
    bulk_transfer_coeff = calculate_bulk_transfer_coefficient(
        cfg, top_cell_is_ice, time, surface_temp
    )
    surface_specific_humidity = calculate_surface_specific_humidity(
        cfg, time, surface_temp
    )
    return (
        REF_AIR_DENSITY
        * LATENT_HEAT_OF_VAPORISATION
        * bulk_transfer_coeff
        * windspeed
        * (ref_specific_humidity - surface_specific_humidity)
    )
