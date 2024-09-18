"""Test that example script still generates the expected simulation configuration
and that the script runs and plots the simulation successfully.

Note we mark the test to run the script main function as slow as it takes a while
for the simulation to run.

To skip this test run pytest -m "not slow"
"""
from pathlib import Path
import pytest
from seaice3p.example import (
    SIMULATION_DIMENSIONAL_PARAMS,
    create_and_save_config,
    main,
)
from seaice3p import Config, DimensionalParams


def test_example_configuration(tmp_path):
    """Test to see if the generated configuration yaml file is the same as the saved
    reference version. If this test fails it means the dictionary of arguments
    SIMULATION_DIMENSIONAL_PARAMS no longer produces the same configuration. Perhaps
    some of the default values have changed.
    """
    REFERENCE_CONFIG_FILE_PATH = Path(__file__).parent / "reference_data/example.yml"
    create_and_save_config(tmp_path, SIMULATION_DIMENSIONAL_PARAMS)
    config_file_path = tmp_path / (SIMULATION_DIMENSIONAL_PARAMS.name + ".yml")
    test_cfg = Config.load(config_file_path)

    reference_cfg = Config.load(REFERENCE_CONFIG_FILE_PATH)
    assert test_cfg == reference_cfg


def test_example_dimensional_configuration(tmp_path):
    """Test to see if the generated dimensional configuration yaml file is the
    same as the saved reference version.
    If this test fails it means the dictionary of arguments
    SIMULATION_DIMENSIONAL_PARAMS no longer produces the same configuration. Perhaps
    some of the default values have changed.
    """
    REFERENCE_CONFIG_FILE_PATH = (
        Path(__file__).parent / "reference_data/example_dimensional.yml"
    )
    create_and_save_config(tmp_path, SIMULATION_DIMENSIONAL_PARAMS)
    config_file_path = tmp_path / (
        SIMULATION_DIMENSIONAL_PARAMS.name + "_dimensional.yml"
    )
    test_cfg = DimensionalParams.load(config_file_path)

    reference_cfg = DimensionalParams.load(REFERENCE_CONFIG_FILE_PATH)
    assert test_cfg == reference_cfg


@pytest.mark.slow
def test_example_script_runs(tmp_path):
    """Check the example script runs with the specified parameters and directories"""
    main(tmp_path, tmp_path, SIMULATION_DIMENSIONAL_PARAMS)
    # Note here we don't need to check that the script ran the simulation as it will
    # crash if it can't find data to plot
