"""Run the simulations as a batch of non dimensional configurations using yearly or
constant forcing options using the commmand line interface.

This simulataneously tests yearly and RED solvers run as a batch and the command line
interface for non-dimensional configurations.
"""

from pathlib import Path
import subprocess
import pytest

TEST_CONFIG_DIR = Path(__file__).parent / "test_configurations/YearlyConstant"


@pytest.mark.slow
def test_running_YearlyConstant_configs():
    subprocess.run(
        f"python -m celestine {TEST_CONFIG_DIR} test_data_YearlyConstant", shell=True
    )
