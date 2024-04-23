from scipy.integrate import solve_ivp
from pathlib import Path
import numpy as np
from celestine.velocities import (
    calculate_velocities,
)
from celestine.brine_channel_sink_terms import (
    calculate_heat_sink,
    calculate_salt_sink,
    calculate_gas_sink,
)
from celestine.state import State, StateBCs, Solution
from celestine.solvers.template import (
    SolverTemplate,
)
import celestine.logging_config as logs


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


class ScipySolver(SolverTemplate):
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

    def ode_fun(self, time, solution_vector):
        print(
            f"{self.cfg.name}: time={time:.3f}/{self.cfg.total_time}\r",
            end="",
        )
        enthalpy, salt, gas = np.split(solution_vector, 3)
        cfg = self.cfg
        D_g = self.D_g
        D_e = self.D_e

        state = State(cfg, time, enthalpy, salt, gas)
        state.calculate_enthalpy_method()
        state_BCs = StateBCs(state)

        Vg, Wl, V = calculate_velocities(state_BCs, cfg)
        Vg = prevent_gas_rise_into_saturated_cell(Vg, state_BCs)

        heat_flux, salt_flux, gas_flux = np.split(
            state_BCs.calculate_fluxes(Wl, Vg, V, D_g), 3
        )

        heat_sink = calculate_heat_sink(state_BCs, cfg)
        salt_sink = calculate_salt_sink(state_BCs, cfg)
        gas_sink = calculate_gas_sink(state_BCs, cfg)

        enthalpy_function = -np.matmul(D_e, heat_flux) - heat_sink
        salt_function = -np.matmul(D_e, salt_flux) - salt_sink
        gas_function = -np.matmul(D_e, gas_flux) - gas_sink

        return np.hstack((enthalpy_function, salt_function, gas_function))

    @logs.time_function
    def solve(self, directory: Path):

        # for the barrow forcing you need to load external data to the forcing config
        self.load_forcing_data_if_needed()

        state = self.generate_initial_solution()
        initial = np.hstack((state.enthalpy, state.salt, state.gas))
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

        sol_enthalpy, sol_salt, sol_gas = np.split(sol.y, 3)

        stored_solution = Solution(self.cfg)
        stored_solution.times = sol.t
        stored_solution.time_length = sol.t.size
        stored_solution.enthalpy = sol_enthalpy
        stored_solution.salt = sol_salt
        stored_solution.gas = sol_gas
        stored_solution.pressure = np.zeros_like(sol_gas)
        stored_solution.save(directory)
        print("")
        return 0
