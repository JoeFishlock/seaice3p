"""Run the simulations as a batch of non dimensional configurations using yearly or
constant forcing options using the commmand line interface.

This simulataneously tests yearly and RED solvers run as a batch and the command line
interface for non-dimensional configurations.
"""

from pathlib import Path
import subprocess
import pytest
from glob import glob

TEST_CONFIG_DIR = Path(__file__).parent / "test_configurations/YearlyConstant"


@pytest.mark.slow
def test_running_YearlyConstant_configs(tmp_path):
    subprocess.run(f"python -m celestine {TEST_CONFIG_DIR} {tmp_path}", shell=True)
    # This command will exit successfully even if some of the simulations crashed
    # test that it produced output for each config
    assert len(glob(str(TEST_CONFIG_DIR / "*.yml"))) == len(
        glob(str(tmp_path / "*.npz"))
    )
