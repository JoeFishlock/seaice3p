from pathlib import Path
from . import (
    DimensionalParams,
    DimensionalEQMGasParams,
    DimensionalMonoBubbleParams,
    DimensionalRJW14Params,
    NoBrineConvection,
    DimensionalRadForcing,
    DimensionalOilInitialConditions,
    NumericalParams,
    DimensionalWaterParams,
    DimensionalConstantSWForcing,
    DimensionalBackgroundOilHeating,
    DimensionalMobileOilHeating,
    DimensionalNoHeating,
    DimensionalConstantTurbulentFlux,
    DimensionalConstantLWForcing,
    DimensionalFixedTempOceanForcing,
)
from .oil_mass import convert_oil_mass_ratio_to_gas_fraction


def generate_oil_simulation_config(
    name: str,
    total_time_in_days: float,
    lengthscale: float,
    initial_oil_mass_ratio: float,
    oil_density: float,
    oil_droplet_radius: float,
    SW_irradiance: float,
    SW_penetration_fraction: float,
    LW_irradiance: float,
    air_temp: float,
    windspeed: float,
    ref_height: float,
    oil_heating_params: DimensionalBackgroundOilHeating
    | DimensionalMobileOilHeating
    | DimensionalNoHeating,
    initial_ice_depth: float,
    initial_ice_temperature: float,
    initial_ocean_temperature: float,
    initial_ice_bulk_salinity: float = 5.92,
    initial_oil_free_ice_depth: float = 0,
    SW_min_wavelength=350,
    SW_max_wavelength=3000,
    num_wavelength_samples=7,
    solver_choice="RK23",
    eddy_diffusivity=0,
    brine_convection_params: DimensionalRJW14Params
    | NoBrineConvection = DimensionalRJW14Params(),
    I=50,
    savefreq_in_days=1.0,
    config_directory=Path("."),
) -> None:
    """Parameters to generate a simulation config for melting of an initially uniform
    layer of ice in an ocean under SW, LW radiative fluxes and sensible heat flux.

    The latent heat flux is disabled by setting the latent heat of vaporisation to 0.

    The initially uniform mass concentration of oil in the domain is set in ng/g.
    """
    ICE_DENSITY = 916
    DimensionalParams(
        name=name,
        total_time_in_days=total_time_in_days,
        savefreq_in_days=savefreq_in_days,
        lengthscale=lengthscale,
        gas_params=DimensionalEQMGasParams(
            gas_density=oil_density,
            saturation_concentration=0,
            tolerable_super_saturation_fraction=1,
            gas_diffusivity=0,
        ),
        bubble_params=DimensionalMonoBubbleParams(
            pore_radius=1.95e-4,
            pore_throat_scaling=0.46,
            porosity_threshold=True,
            porosity_threshold_value=0.024,
            bubble_radius=oil_droplet_radius,
            escape_ice_surface=False,
        ),
        brine_convection_params=brine_convection_params,
        forcing_config=DimensionalRadForcing(
            SW_forcing=DimensionalConstantSWForcing(
                SW_irradiance=SW_irradiance,
                SW_min_wavelength=SW_min_wavelength,
                SW_max_wavelength=SW_max_wavelength,
                num_wavelength_samples=num_wavelength_samples,
                SW_penetration_fraction=SW_penetration_fraction,
            ),
            LW_forcing=DimensionalConstantLWForcing(LW_irradiance=LW_irradiance),
            turbulent_flux=DimensionalConstantTurbulentFlux(
                ref_height=ref_height,
                windspeed=windspeed,
                air_temp=air_temp,
                air_latent_heat_of_vaporisation=0,
            ),
            oil_heating=oil_heating_params,
        ),
        ocean_forcing_config=DimensionalFixedTempOceanForcing(
            ocean_temp=initial_ocean_temperature
        ),
        initial_conditions_config=DimensionalOilInitialConditions(
            initial_ice_depth=initial_ice_depth,
            initial_ice_bulk_salinity=initial_ice_bulk_salinity,
            initial_ice_temperature=initial_ice_temperature,
            initial_ocean_temperature=initial_ocean_temperature,
            initial_oil_volume_fraction=convert_oil_mass_ratio_to_gas_fraction(
                initial_oil_mass_ratio, oil_density, ice_density=ICE_DENSITY
            ),
            initial_oil_free_depth=initial_oil_free_ice_depth,
        ),
        water_params=DimensionalWaterParams(
            liquid_density=1028,
            ice_density=ICE_DENSITY,
            ocean_salinity=34,
            salt_diffusivity=0,
            eddy_diffusivity=eddy_diffusivity,
            liquid_thermal_conductivity=0.54,
        ),
        numerical_params=NumericalParams(I=I, solver_choice=solver_choice),
        frame_velocity_dimensional=0,
    ).save(config_directory)
