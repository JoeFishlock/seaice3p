from celestine.velocities import (
    calculate_velocities,
)
from celestine.flux import (
    calculate_heat_flux,
    calculate_salt_flux,
    calculate_gas_flux,
    take_forward_euler_step,
)
from celestine.state import State, StateBCs
from celestine.solvers.template import (
    SolverTemplate,
    prevent_gas_rise_into_saturated_cell,
)


class ReducedSolver(SolverTemplate):
    """Take timestep using forward Euler upwind scheme using reduced model."""

    def pre_solve_checks(self):
        self.cfg.check_thermal_Courant_number()

    def take_timestep(self, state: State):
        cfg = self.cfg
        timestep = cfg.numerical_params.timestep

        D_g = self.D_g
        D_e = self.D_e

        time = state.time
        new_time = time + timestep

        # calculate temperature, salinity etc for state on grid centers
        state.calculate_enthalpy_method()
        state_BCs = StateBCs(state)  # Add boundary conditions

        Vg, Wl, V = calculate_velocities(state_BCs, cfg)
        Vg = prevent_gas_rise_into_saturated_cell(Vg, state_BCs)

        heat_flux = calculate_heat_flux(state_BCs, Wl, V, D_g, cfg)
        salt_flux = calculate_salt_flux(state_BCs, Wl, V, D_g, cfg)
        gas_flux = calculate_gas_flux(state_BCs, Wl, V, Vg, D_g, cfg)

        new_enthalpy = take_forward_euler_step(state.enthalpy, heat_flux, timestep, D_e)
        new_salt = take_forward_euler_step(state.salt, salt_flux, timestep, D_e)
        new_gas = take_forward_euler_step(state.gas, gas_flux, timestep, D_e)

        new_state = State(
            cfg,
            new_time,
            new_enthalpy,
            new_salt,
            new_gas,
        )
        return new_state
