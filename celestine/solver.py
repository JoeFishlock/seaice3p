from scipy.integrate import solve_ivp
from pathlib import Path
import numpy as np
from celestine.velocities import (
    calculate_velocities,
)
from celestine.state import EQMState, StateBCs
import celestine.logging_config as logs
from .params import Config
from .grids import get_difference_matrix
from .initial_conditions import get_initial_conditions


def prevent_gas_rise_into_saturated_cell(Vg, state_BCs: StateBCs):
    """Modify the gas interstitial velocity to prevent bubble rise into a cell which
    is already theoretically saturated with gas.

    From the state with boundary conditions calculate the gas and solid fraction in the
    cells (except at lower ghost cell). If any of these are such that there is more gas
    fraction than pore space available then set gas insterstitial velocity to zero on
    the edge below. Make sure the very top boundary velocity is not changed as we want
    to always alow flux to the atmosphere regardless of the boundary conditions imposed.

    :param Vg: gas insterstitial velocity on cell edges
    :type Vg: Numpy array (size I+1)
    :param state_BCs: state of system with boundary conditions
    :type state_BCs: celestine.state.StateBCs
    :return: filtered gas interstitial velocities on edges to prevent gas rise into a
        fully gas saturated cell

    """
    gas_fraction_above = state_BCs.gas_fraction[1:]
    solid_fraction_above = 1 - state_BCs.liquid_fraction[1:]
    filtered_Vg = np.where(gas_fraction_above + solid_fraction_above >= 1, 0, Vg)
    filtered_Vg[-1] = Vg[-1]
    return filtered_Vg


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
        self.D_e = get_difference_matrix(self.I, self.step)
        self.D_g = get_difference_matrix(self.I + 1, self.step)

    @property
    def number_of_solution_components(self):
        """This determines how many components the solution object is split into when
        saved and therefore must be determined from the configuraiton to match the
        state object"""
        return 3

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
        cfg = self.cfg
        D_g = self.D_g
        D_e = self.D_e

        state = EQMState.init_from_stacked_state(cfg, time, solution_vector)
        state.calculate_enthalpy_method()
        state_BCs = StateBCs(state)

        Vg, Wl, V = calculate_velocities(state_BCs, cfg)
        Vg = prevent_gas_rise_into_saturated_cell(Vg, state_BCs)

        return (
            -state_BCs.calculate_dz_fluxes(Wl, Vg, V, D_g, D_e)
            - state_BCs.calculate_brine_convection_sink()
        )

    @logs.time_function
    def solve(self, directory: Path):

        self.pre_solve_checks()

        # for the barrow forcing you need to load external data to the forcing config
        self.load_forcing_data_if_needed()

        initial = get_initial_conditions(self.cfg).get_stacked_state()
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
