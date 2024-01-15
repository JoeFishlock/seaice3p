"""Load and run all simulation configuration yaml files in test_data/ and save data.

log if the simulation runs or crashes.
"""
from glob import glob
from celestine.params import Config
from celestine.run_simulation import run_batch
from celestine.logging_config import logger
from celestine.__init__ import __version__
from tests.generate_tests import TEST_DATA_DIR

if __name__ == "__main__":
    logger.info(f"Celestine version {__version__}")
    logger.info(f"Running test simulations in {TEST_DATA_DIR}")

    config_paths = glob(f"{TEST_DATA_DIR}*.yml")
    configs = [Config.load(path) for path in config_paths]
    run_batch(configs, TEST_DATA_DIR)
