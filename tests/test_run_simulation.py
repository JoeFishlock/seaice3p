import pytest
from celestine import solve
from celestine import DimensionalParams


@pytest.mark.parametrize(
    "simulation_parameters",
    [
        {
            "name": "defaults",
            "total_time_in_days": 1,
            "savefreq_in_days": 0.1,
        },
        {
            "name": "barrow_forcing",
            "total_time_in_days": 1,
            "savefreq_in_days": 0.1,
            "temperature_forcing_choice": "barrow_2009",
            "Barrow_top_temperature_data_choice": "top_ice",
        },
        # {
        #     # This will fail because of the coupling to horizontal flow is True
        #     "name": "barrow_brine_drainage_defaults",
        #     "total_time_in_days": 1,
        #     "savefreq_in_days": 0.1,
        #     "temperature_forcing_choice": "barrow_2009",
        #     "Barrow_top_temperature_data_choice": "top_ice",
        #     "brine_convection_parameterisation": True,
        # },
        {
            "name": "barrow_brine_drainage_no_horizontal_eqm",
            "total_time_in_days": 1,
            "savefreq_in_days": 0.1,
            "temperature_forcing_choice": "barrow_2009",
            "Barrow_top_temperature_data_choice": "top_ice",
            "brine_convection_parameterisation": True,
            "couple_bubble_to_horizontal_flow": False,
        },
        {
            "name": "barrow_brine_drainage_no_horizontal_diseq",
            "total_time_in_days": 1,
            "savefreq_in_days": 0.1,
            "temperature_forcing_choice": "barrow_2009",
            "Barrow_top_temperature_data_choice": "top_ice",
            "brine_convection_parameterisation": True,
            "couple_bubble_to_horizontal_flow": False,
            "model": "DISEQ",
        },
    ],
)
def test_short_solve(tmp_path, simulation_parameters):
    print(tmp_path)
    solve(DimensionalParams(**simulation_parameters).get_config(), tmp_path)
