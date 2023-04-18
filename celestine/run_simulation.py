from celestine.params import Config
from celestine.logging_config import logger
from celestine.solvers.lax_friedrich_solver import LXFSolver
from celestine.solvers.lagged_solver import LaggedUpwindSolver


def solve(cfg: Config):
    SOLVER_OPTIONS = {"LXF": LXFSolver, "LU": LaggedUpwindSolver}
    solver_choice = cfg.numerical_params.solver
    if solver_choice in SOLVER_OPTIONS.keys():
        solver_class = SOLVER_OPTIONS[solver_choice]
        solver_instance = solver_class(cfg)
        return solver_instance.solve()
    else:
        logger.error(
            f"config {cfg.name} solver choice {solver_choice} is not an option"
        )
        raise KeyError(f"solver choice {solver_choice} is not an option")
