"""Module to run the simulation on the given configuration with the appropriate solver.

Solve reduced model using scipy solve_ivp using RK23 solver.

Impose a maximum timestep constraint using courant number for thermal diffusion
as this is an explicit method.

This solver uses adaptive timestepping which makes it a good choice for running
simulations with large buoyancy driven gas bubble velocities and we save the output
at intervals given by the savefreq parameter in configuration.
"""
from pathlib import Path
from typing import Literal, Callable, List
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .printing import get_printer
from .equations import get_equations
from .state import get_unpacker
from .forcing import get_boundary_conditions
from .enthalpy_method import get_enthalpy_method
from .params import Config, EQMPhysicalParams, DISEQPhysicalParams
from .grids import Grids
from .initial_conditions import get_initial_conditions


def run_batch(list_of_cfg: List[Config], directory: Path, verbosity_level=0) -> None:
    """Run a batch of simulations from a list of configurations.

    Each simulation name is logged, as well as if it successfully runs or crashes.
    Output from each simulation is saved in a .npz file.

    :param list_of_cfg: list of configurations
    :type list_of_cfg: List[seaice3p.params.Config]

    """
    optprint = get_printer(verbosity_level)
    for cfg in list_of_cfg:
        try:
            solve(cfg, directory, verbosity_level=verbosity_level)
        except Exception as e:
            optprint(f"{cfg.name} crashed")
            optprint(f"{e}")


def solve(cfg: Config, directory: Path, verbosity_level=0) -> Literal[0]:
    if isinstance(cfg.physical_params, EQMPhysicalParams):
        number_of_solution_components = 3
    elif isinstance(cfg.physical_params, DISEQPhysicalParams):
        number_of_solution_components = 4
    else:
        raise NotImplementedError

    initial = get_initial_conditions(cfg)
    T = cfg.total_time
    t_eval = np.arange(0, T, cfg.savefreq)
    ode_fun = _get_ode_fun(cfg, verbosity_level=verbosity_level)

    if cfg.numerical_params.solver_choice in ["RK23", "RK45", "DOP853"]:
        # Explicit method so set courant timestep limit
        max_diffusivity = max(
            cfg.physical_params.conductivity_ratio,
            1 + cfg.physical_params.eddy_diffusivity_ratio,
        )
        max_step = 0.45 * (1 / max_diffusivity) * cfg.numerical_params.step**2
    else:
        # Implicit method no timestep restriction
        max_step = np.inf

    sol = solve_ivp(
        ode_fun,
        [0, T],
        initial,
        t_eval=t_eval,
        max_step=max_step,
        method=cfg.numerical_params.solver_choice,
    )

    # Note that to keep the solution components general we must just save with
    # defaults so that time corresponds to "arr_0", next component "arr_1" etc...
    np.savez(
        directory / f"{cfg.name}.npz",
        sol.t,
        *np.split(sol.y, number_of_solution_components),
    )
    optprint = get_printer(verbosity_level)
    optprint("")
    return 0


def _get_ode_fun(cfg: Config, verbosity_level=0) -> Callable[[float, NDArray], NDArray]:

    grids = Grids(cfg.numerical_params.I)
    enthalpy_method = get_enthalpy_method(cfg)
    boundary_conditions = get_boundary_conditions(cfg)
    unpack = get_unpacker(cfg)
    equations = get_equations(cfg, grids)

    optprint = get_printer(verbosity_level)

    def ode_fun(time, solution_vector):
        optprint(
            f"{cfg.name}: time={time:.3f}/{cfg.total_time}\r",
            end="",
        )

        # Let state module handle providing the correct State class based on
        # simulation configuration
        state = unpack(time, solution_vector)
        full_state = enthalpy_method(state)
        state_BCs = boundary_conditions(full_state)

        return equations(state_BCs)

    return ode_fun
