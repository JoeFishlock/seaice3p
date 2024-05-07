"""Test running the diagnostic scripts in celestine

These scripts plot functions used in the code like the gas velocities and brine
convection parameteristaion

These are useful for making plots and also as integration tests for these parts of
the code however they do produce file output
"""
import pytest


@pytest.mark.slow
def test_running_gas_velocity(tmp_path):
    from celestine.diagnostics.gas_velocity import main

    main(tmp_path)


@pytest.mark.slow
def test_running_shortwave_case(tmp_path):
    from celestine.diagnostics.shortwave_case import main, SIMULATION_DIMENSIONAL_PARAMS

    main(tmp_path, tmp_path, SIMULATION_DIMENSIONAL_PARAMS)


def test_running_brine_convection_parameterisation(tmp_path):
    from celestine.diagnostics.brine_drainage_parameterisation import main

    main(tmp_path)
