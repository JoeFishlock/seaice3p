"""
Test running best configuration for barrow configuration with brine convection
and power law gas bubble size distribution.

Run using the CLI interface with the single configuration and dimensional options
"""

from pathlib import Path
import subprocess
import pytest

TEST_CONFIG_PATH = (
    Path(__file__).parent / "test_configurations/best_barrow_dimensional.yml"
)


@pytest.mark.slow
def test_running_best_Barrow():
    subprocess.run(
        f"python -m celestine --single {TEST_CONFIG_PATH} test_data_best_barrow --dimensional",
        shell=True,
    )
