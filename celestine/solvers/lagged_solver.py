import numpy as np
from celestine.velocities import (
    calculate_velocities,
    calculate_absolute_permeability,
    solve_pressure_equation,
)
from celestine.flux import calculate_heat_flux, calculate_salt_flux, calculate_gas_flux
from celestine.solvers.template import SolverTemplate, State, StateBCs


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
        I = self.I
        timestep = cfg.numerical_params.timestep

        D_g = self.D_g
        D_e = self.D_e

        time = state.time
        new_time = time + timestep

        # calculate temperature, salinity etc for state on grid centers
        state.calculate_enthalpy_method(cfg)

        state_BCs = StateBCs(state, cfg)

        Vg, Wl, V = calculate_velocities(
            state.liquid_fraction, state.pressure, D_g, cfg
        )

        new_pressure = np.zeros((I,))

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
        gas_fraction = state.gas_fraction

        new_liquid_fraction = new_state.liquid_fraction
        new_gas_fraction = new_state.gas_fraction
        new_permeability = calculate_absolute_permeability(new_liquid_fraction)

        new_pressure = solve_pressure_equation(
            gas_fraction, new_gas_fraction, new_permeability, timestep, D_e, D_g, cfg
        )
        new_state.pressure = new_pressure

        return new_state
