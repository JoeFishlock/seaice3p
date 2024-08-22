from scipy.integrate import solve_ivp
from pathlib import Path
import numpy as np

from .equations import calculate_equations
from .state import get_state, apply_boundary_conditions
from .enthalpy_method import calculate_enthalpy_method
import celestine.logging_config as logs
from .params import Config
from .grids import Grids
from .initial_conditions import get_initial_conditions


class Solver:
    """Solve reduced model using scipy solve_ivp using RK23 solver. This is the "SCI"
    solver option.

    Impose a maximum timestep constraint using courant number for thermal diffusion
    as this is an explicit method.

    This solver uses adaptive timestepping which makes it a good choice for running
    simulations with large buoyancy driven gas bubble velocities and we save the output
    at intervals given by the savefreq parameter in configuration.

    The interface of this class is a little different as we overwrite the solve method
    from the template and must provide a function to calculate the ode forcing for
    solve_ivp. However the solve function still saves the data in the same format using
    the `celestine.state.Solution` class.
    """

    # For explicit heat diffusion stability we require timestep < 0.5 * step^2
    # In the case of enhanced conduction in solid we multiply by
    # (liquid_fraction * conductivity_ratio*solid_fraction)
    # For typical sea ice parameters reducing the Courant coefficient for stability
    # to 0.1 should suffice.
    THERMAL_DIFFUSION_TIMESTEP_LIMIT = 0.1

    def __init__(self, cfg: Config):
        """initialise solver object

        Assign step size, number of cells and difference matrices for convenience.

        :param cfg: simulation configuration
        """
        self.cfg = cfg
        self.step = cfg.numerical_params.step
        self.I = cfg.numerical_params.I
        self.D_e = Grids(cfg.numerical_params.I).D_e
        self.D_g = Grids(cfg.numerical_params.I).D_g
        self.grids = Grids(cfg.numerical_params.I)

    @property
    def number_of_solution_components(self):
        """This determines how many components the solution object is split into when
        saved and therefore must be determined from the configuraiton to match the
        state object"""
        STATE_COMPONENTS = {"EQM": 3, "DISEQ": 4}
        return STATE_COMPONENTS[self.cfg.model]

    def pre_solve_checks(self):
        """Optionally implement this method if you want to check anything before
        running the solver.

        For example to check the timestep and grid step satisfy some constraint.
        """
        pass

    def load_forcing_data_if_needed(self):
        if self.cfg.forcing_config.temperature_forcing_choice == "barrow_2009":
            self.cfg.forcing_config.load_forcing_data()

    def ode_fun(self, time, solution_vector):
        print(
            f"{self.cfg.name}: time={time:.3f}/{self.cfg.total_time}\r",
            end="",
        )

        # Let state module handle providing the correct State class based on
        # simulation configuration
        state = get_state(self.cfg, time, solution_vector)
        full_state = calculate_enthalpy_method(self.cfg, state)
        state_BCs = apply_boundary_conditions(self.cfg, full_state)

        return calculate_equations(state_BCs, self.cfg, self.grids)

    @logs.time_function
    def solve(self, directory: Path):

        self.pre_solve_checks()

        # for the barrow forcing you need to load external data to the forcing config
        self.load_forcing_data_if_needed()

        initial = get_initial_conditions(self.cfg)
        T = self.cfg.total_time
        t_eval = np.arange(0, T, self.cfg.savefreq)

        sol = solve_ivp(
            self.ode_fun,
            [0, T],
            initial,
            t_eval=t_eval,
            max_step=self.THERMAL_DIFFUSION_TIMESTEP_LIMIT
            * self.cfg.numerical_params.step**2,
            method="RK23",
        )

        # Note that to keep the solution components general we must just save with
        # defaults so that time corresponds to "arr_0", next component "arr_1" etc...
        np.savez(
            directory / f"{self.cfg.name}.npz",
            sol.t,
            *np.split(sol.y, self.number_of_solution_components),
        )
        print("")
        return 0
