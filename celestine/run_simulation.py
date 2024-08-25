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


from .equations import get_equations
from .state import get_unpacker
from .forcing import get_boundary_conditions
from .enthalpy_method import get_enthalpy_method
from .params import Config
from .grids import Grids
from .initial_conditions import get_initial_conditions


def run_batch(list_of_cfg: List[Config], directory: Path) -> None:
    """Run a batch of simulations from a list of configurations.

    Each simulation name is logged, as well as if it successfully runs or crashes.
    Output from each simulation is saved in a .npz file.

    :param list_of_cfg: list of configurations
    :type list_of_cfg: List[celestine.params.Config]

    """
    for cfg in list_of_cfg:
        try:
            solve(cfg, directory)
        except Exception as e:
            print(f"{cfg.name} crashed")
            print(f"{e}")


# For explicit heat diffusion stability we require timestep < 0.5 * step^2
# In the case of enhanced conduction in solid we multiply by
# (liquid_fraction * conductivity_ratio*solid_fraction)
# For typical sea ice parameters reducing the Courant coefficient for stability
# to 0.1 should suffice.
THERMAL_DIFFUSION_TIMESTEP_LIMIT = 0.1


def solve(cfg: Config, directory: Path) -> Literal[0]:
    if cfg.model == "EQM":
        number_of_solution_components = 3
    elif cfg.model == "DISEQ":
        number_of_solution_components = 4
    else:
        raise NotImplementedError

    initial = get_initial_conditions(cfg)
    T = cfg.total_time
    t_eval = np.arange(0, T, cfg.savefreq)
    ode_fun = _get_ode_fun(cfg)

    sol = solve_ivp(
        ode_fun,
        [0, T],
        initial,
        t_eval=t_eval,
        max_step=THERMAL_DIFFUSION_TIMESTEP_LIMIT * cfg.numerical_params.step**2,
        method="RK23",
    )

    # Note that to keep the solution components general we must just save with
    # defaults so that time corresponds to "arr_0", next component "arr_1" etc...
    np.savez(
        directory / f"{cfg.name}.npz",
        sol.t,
        *np.split(sol.y, number_of_solution_components),
    )
    print("")
    return 0


def _get_ode_fun(cfg: Config) -> Callable[[float, NDArray], NDArray]:

    grids = Grids(cfg.numerical_params.I)
    enthalpy_method = get_enthalpy_method(cfg)
    boundary_conditions = get_boundary_conditions(cfg)
    unpack = get_unpacker(cfg)
    equations = get_equations(cfg, grids)

    def ode_fun(time, solution_vector):
        print(
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
