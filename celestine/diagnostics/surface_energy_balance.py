"""Script to run a simulation starting with summer initial condition with surface
energy balance top forcing.

For now adjust hard coded in values to see if this is working as expected
"""

from pathlib import Path
from celestine.example import main

DATA_DIRECTORY = Path("surface_energy_balance_test")
FRAMES_DIR = Path("surface_energy_balance_test/frames")
SIMULATION_DIMENSIONAL_PARAMS = {
    "name": "surface-energy-balance-test",
    "SW_internal_heating": False,
    "surface_energy_balance_forcing": True,
    "total_time_in_days": 90,
    "savefreq_in_days": 1,
    "lengthscale": 2.4,
    "I": 30,
    "phase_average_conductivity": True,
    "brine_convection_parameterisation": True,
    "couple_bubble_to_horizontal_flow": False,
    "couple_bubble_to_vertical_flow": False,
    "Rayleigh_critical": 2.9,
    "convection_strength": 0.13,
    "initial_conditions_choice": "summer",
    "initial_summer_ice_depth": 1,
    "initial_summer_ocean_temperature": -2,
    "far_temp": -2,
    "initial_summer_ice_temperature": -4,
    "far_gas_sat": 0,
    "temperature_forcing_choice": "constant",
    "constant_top_temperature": 0,
    "SW_forcing_choice": "constant",
    "constant_SW_irradiance": 0,
    "SW_radiation_model_choice": "1L",
    "constant_oil_mass_ratio": 1000,
    "SW_scattering_ice_type": "FYI",
}


if __name__ == "__main__":
    main(DATA_DIRECTORY, FRAMES_DIR, SIMULATION_DIMENSIONAL_PARAMS)
