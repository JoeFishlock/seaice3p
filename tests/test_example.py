"""Test that example script still generates the expected simulation configuration
and that the script runs and plots the simulation successfully.

Note we mark the test to run the script main function as slow as it takes a while
for the simulation to run.

To skip this test run pytest -m "not slow"
"""
from pathlib import Path
import pytest
from tempfile import TemporaryDirectory
from celestine.example import (
    SIMULATION_DIMENSIONAL_PARAMS,
    DATA_DIRECTORY,
    FRAMES_DIR,
    create_and_save_config,
    main,
)
from celestine.params import Config
from celestine.dimensional_params import DimensionalParams


def test_example_configuration():
    """Test to see if the generated configuration yaml file is the same as the saved
    reference version. If this test fails it means the dictionary of arguments
    SIMULATION_DIMENSIONAL_PARAMS no longer produces the same configuration. Perhaps
    some of the default values have changed.
    """
    REFERENCE_CONFIG_FILE_PATH = Path(__file__).parent / "reference_data/example.yml"
    with TemporaryDirectory(dir=".") as Temporary_directory_path:
        Temporary_directory_path = Path(Temporary_directory_path)
        create_and_save_config(Temporary_directory_path, SIMULATION_DIMENSIONAL_PARAMS)
        config_file_path = Temporary_directory_path / (
            SIMULATION_DIMENSIONAL_PARAMS["name"] + ".yml"
        )
        test_cfg = Config.load(config_file_path)

    reference_cfg = Config.load(REFERENCE_CONFIG_FILE_PATH)
    assert test_cfg == reference_cfg


def test_example_dimensional_configuration():
    """Test to see if the generated dimensional configuration yaml file is the
    same as the saved reference version.
    If this test fails it means the dictionary of arguments
    SIMULATION_DIMENSIONAL_PARAMS no longer produces the same configuration. Perhaps
    some of the default values have changed.
    """
    REFERENCE_CONFIG_FILE_PATH = (
        Path(__file__).parent / "reference_data/example_dimensional.yml"
    )
    with TemporaryDirectory(dir=".") as Temporary_directory_path:
        Temporary_directory_path = Path(Temporary_directory_path)
        create_and_save_config(Temporary_directory_path, SIMULATION_DIMENSIONAL_PARAMS)
        config_file_path = Temporary_directory_path / (
            SIMULATION_DIMENSIONAL_PARAMS["name"] + "_dimensional.yml"
        )
        test_cfg = DimensionalParams.load(config_file_path)

    reference_cfg = DimensionalParams.load(REFERENCE_CONFIG_FILE_PATH)
    assert test_cfg == reference_cfg


@pytest.mark.slow
def test_example_script_runs():
    """Check the example script runs with the specified parameters and directories"""
    main(DATA_DIRECTORY, FRAMES_DIR, SIMULATION_DIMENSIONAL_PARAMS)
