"""Module to run the simulation on the given configuration with the appropriate solver.
"""
from pathlib import Path
from celestine.params import Config
from celestine.logging_config import logger, log_time
from .solver import Solver


def solve(cfg: Config, directory: Path):
    """Solve simulation choosing appropriate solver from the choice in the config."""

    return Solver(cfg).solve(directory)


def run_batch(list_of_cfg, directory: Path):
    """Run a batch of simulations from a list of configurations.

    Each simulation name is logged, as well as if it successfully runs or crashes.
    Output from each simulation is saved in a .npz file.

    :param list_of_cfg: list of configurations
    :type list_of_cfg: List[celestine.params.Config]

    """
    for cfg in list_of_cfg:
        logger.info(f"Running {cfg.name}")
        try:
            _, duration = solve(cfg, directory)
            log_time(logger, duration, message=f"{cfg.name} ran in ")
        except Exception as e:
            logger.error(f"{cfg.name} crashed")
            logger.error(f"{e}")
