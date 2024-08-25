from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dimensional import DimensionalParams

from .forcing import (
    ForcingConfig,
    ConstantForcing,
    BRW09Forcing,
    YearlyForcing,
    RadForcing,
)


def get_dimensionless_forcing_config(
    dimensional_params: "DimensionalParams",
) -> ForcingConfig:
    match dimensional_params.temperature_forcing_choice:
        case "constant":
            top_temp = (
                dimensional_params.constant_top_temperature
                - dimensional_params.ocean_freezing_temperature
            ) / dimensional_params.temperature_difference
            return ConstantForcing(constant_top_temperature=top_temp)
        case "yearly":
            return YearlyForcing(
                offset=dimensional_params.offset,
                amplitude=dimensional_params.amplitude,
                period=dimensional_params.period,
            )
        case "barrow_2009":
            return BRW09Forcing(
                Barrow_top_temperature_data_choice=dimensional_params.Barrow_top_temperature_data_choice,
                Barrow_initial_bulk_gas_in_ice=dimensional_params.Barrow_initial_bulk_gas_in_ice,
            )
        case "rad":
            return RadForcing(
                surface_energy_balance_forcing=dimensional_params.surface_energy_balance_forcing,
                SW_internal_heating=dimensional_params.SW_internal_heating,
                SW_forcing_choice=dimensional_params.SW_forcing_choice,
                constant_SW_irradiance=dimensional_params.constant_SW_irradiance,
                SW_radiation_model_choice=dimensional_params.SW_radiation_model_choice,
                constant_oil_mass_ratio=dimensional_params.constant_oil_mass_ratio,
                SW_scattering_ice_type=dimensional_params.SW_scattering_ice_type,
            )
        case _:
            raise NotImplementedError
