from importlib import import_module
from celestine.params import Config
from celestine.logging_config import logger


def solve(cfg: Config):
    SOLVER_OPTIONS = {
        "LXF": "lax_friedrich_solver",
    }
    solver_choice = cfg.numerical_params.solver
    if solver_choice in SOLVER_OPTIONS.keys():
        solver_module_name = SOLVER_OPTIONS[solver_choice]
        solver = import_module(f"celestine.solvers.{solver_module_name}")
    else:
        logger.error(
            f"config {cfg.name} solver choice {solver_choice} is not an option"
        )
        raise KeyError(f"solver choice {solver_choice} is not an option")

    return solver.solve(cfg)
