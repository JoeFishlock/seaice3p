"""Load and run all simulation configuration yaml files in test_data/ and save data.

log if the simulation runs or crashes.
"""
from glob import glob
from celestine.params import Config
from celestine.run_simulation import solve
from celestine.logging_config import logger, log_time
from celestine.__init__ import __version__

logger.info(f"Celestine version {__version__}")

test_path = "test_data/"

logger.info(f"Running test simulations in {test_path}")

config_paths = glob(f"{test_path}*.yml")
for path in config_paths:
    cfg = Config.load(path)
    logger.info(f"Running {cfg.name}")
    try:
        status, duration = solve(cfg)
        log_time(logger, duration, message=f"{cfg.name} ran in ")
    except Exception as e:
        logger.error(f"{cfg.name} crashed")
        logger.error(f"{e}")
