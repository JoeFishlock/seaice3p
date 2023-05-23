"""Module to run the simulation on the given configuration with the appropriate solver.
"""

from celestine.params import Config
from celestine.logging_config import logger, log_time
from celestine.solvers.lagged_solver import LaggedUpwindSolver
from celestine.solvers.reduced_solver import ReducedSolver
from celestine.solvers.scipy import ScipySolver


def solve(cfg: Config):
    """Solve simulation choosing appropriate solver from the choice in the config."""
    SOLVER_OPTIONS = {
        "LU": LaggedUpwindSolver,
        "RED": ReducedSolver,
        "SCI": ScipySolver,
    }
    solver_choice = cfg.numerical_params.solver
    if solver_choice in SOLVER_OPTIONS.keys():
        solver_class = SOLVER_OPTIONS[solver_choice]
        solver_instance = solver_class(cfg)
        return solver_instance.solve()

    logger.error(f"config {cfg.name} solver choice {solver_choice} is not an option")
    raise KeyError(f"solver choice {solver_choice} is not an option")


def run_batch(list_of_cfg):
    """Run a batch of simulations from a list of configurations.

    Each simulation name is logged, as well as if it successfully runs or crashes.
    Output from each simulation is saved in a .npz file.

    :param list_of_cfg: list of configurations
    :type list_of_cfg: List[celestine.params.Config]

    """
    for cfg in list_of_cfg:
        logger.info(f"Running {cfg.name}")
        try:
            status, duration = solve(cfg)
            log_time(logger, duration, message=f"{cfg.name} ran in ")
        except Exception as e:
            logger.error(f"{cfg.name} crashed")
            logger.error(f"{e}")
