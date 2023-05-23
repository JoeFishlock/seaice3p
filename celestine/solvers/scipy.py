from scipy.integrate import solve_ivp
import numpy as np
from celestine.velocities import (
    calculate_velocities,
)
from celestine.flux import (
    calculate_heat_flux,
    calculate_salt_flux,
    calculate_gas_flux,
)
from celestine.state import State, StateBCs, Solution
from celestine.solvers.template import SolverTemplate
from celestine.solvers.reduced_solver import prevent_gas_rise_into_saturated_cell
import celestine.logging_config as logs


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

    def take_timestep(self, state: State):
        pass

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

        Vg, Wl, V = calculate_velocities(state_BCs, D_g, cfg)
        Vg = prevent_gas_rise_into_saturated_cell(Vg, state_BCs)

        heat_flux = calculate_heat_flux(state_BCs, Wl, V, D_g)
        salt_flux = calculate_salt_flux(state_BCs, Wl, V, D_g, cfg)
        gas_flux = calculate_gas_flux(state_BCs, Wl, V, Vg, D_g, cfg)

        enthalpy_function = -np.matmul(D_e, heat_flux)
        salt_function = -np.matmul(D_e, salt_flux)
        gas_function = -np.matmul(D_e, gas_flux)

        return np.hstack((enthalpy_function, salt_function, gas_function))

    @logs.time_function
    def solve(self):

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
            max_step=0.4 * self.cfg.numerical_params.step**2,
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
        stored_solution.save()
        print("")
        return 0
