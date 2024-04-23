"""
Test running best configuration for barrow configuration with brine convection
and power law gas bubble size distribution.

Run using the CLI with dimensional option
"""

from pathlib import Path
import subprocess
import pytest

TEST_CONFIG_PATH = Path(__file__).parent / "test_configurations/best_barrow"


@pytest.mark.slow
def test_running_best_Barrow(tmp_path):
    subprocess.run(
        f"python -m celestine {TEST_CONFIG_PATH} {tmp_path} --dimensional",
        shell=True,
    )
    # Test that the simulation has actually run and saved output
    output_file = tmp_path / "best_barrow.npz"
    assert output_file.is_file()
