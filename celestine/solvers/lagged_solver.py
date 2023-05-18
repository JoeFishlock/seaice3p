import numpy as np
from celestine.velocities import (
    calculate_velocities,
    solve_pressure_equation,
)
from celestine.flux import calculate_heat_flux, calculate_salt_flux, calculate_gas_flux
from celestine.state import State, StateBCs
from celestine.solvers.template import SolverTemplate


def take_forward_euler_step(quantity, flux, timestep, D_e):
    r"""Advance the given quantity one forward Euler step using the given flux

    The quantity is given on cell centers and the flux on cell edges.

    Discretise the conservation equation

    .. math:: \frac{\partial Q}{\partial t} = -\frac{\partial F}{\partial z}

    as

    .. math:: Q^{n+1} = Q^n - \Delta t (\frac{\partial F}{\partial z})

    """
    return quantity - timestep * np.matmul(D_e, flux)


class LaggedUpwindSolver(SolverTemplate):
    """Take timestep using upwind scheme with liquid velocity calculation lagged."""

    def take_timestep(self, state: State):
        cfg = self.cfg
        timestep = cfg.numerical_params.timestep

        D_g = self.D_g
        D_e = self.D_e

        time = state.time
        new_time = time + timestep

        # calculate temperature, salinity etc for state on grid centers
        state.calculate_enthalpy_method(cfg)
        state_BCs = StateBCs(state, cfg)  # Add boundary conditions

        Vg, Wl, V = calculate_velocities(state_BCs, D_g, cfg)

        heat_flux = calculate_heat_flux(state_BCs, Wl, V, D_g)
        salt_flux = calculate_salt_flux(state_BCs, Wl, V, D_g, cfg)
        gas_flux = calculate_gas_flux(state_BCs, Wl, V, Vg, D_g, cfg)

        new_enthalpy = take_forward_euler_step(state.enthalpy, heat_flux, timestep, D_e)
        new_salt = take_forward_euler_step(state.salt, salt_flux, timestep, D_e)
        new_gas = take_forward_euler_step(state.gas, gas_flux, timestep, D_e)

        new_state = State(
            new_time,
            new_enthalpy,
            new_salt,
            new_gas,
        )
        new_state.calculate_enthalpy_method(cfg)
        new_state_BCs = StateBCs(new_state, cfg)
        new_pressure = solve_pressure_equation(
            state_BCs, new_state_BCs, timestep, D_e, D_g, cfg
        )
        new_state.pressure = new_pressure[1:-1]

        return new_state
