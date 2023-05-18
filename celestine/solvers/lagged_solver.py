import numpy as np
from celestine.grids import (
    upwind,
    centers_to_edges,
)
from celestine.velocities import (
    calculate_velocities,
    calculate_absolute_permeability,
    solve_pressure_equation,
)
from celestine.flux import calculate_heat_flux
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

        chi = cfg.physical_params.expansion_coefficient
        C = cfg.physical_params.concentration_ratio

        D_g = self.D_g
        D_e = self.D_e

        time = state.time
        new_time = time + timestep

        # calculate temperature, salinity etc for state on grid centers
        state.calculate_enthalpy_method(cfg)
        pressure = state.pressure
        liquid_fraction = state.liquid_fraction
        gas_fraction = state.gas_fraction

        state_BCs = StateBCs(state, cfg)

        dissolved_gas_ghosts = state_BCs.dissolved_gas
        gas_fraction_ghosts = state_BCs.gas_fraction
        gas_ghosts = state_BCs.gas
        liquid_salinity_ghosts = state_BCs.liquid_salinity
        salt_ghosts = state_BCs.salt

        liquid_fraction_edges = centers_to_edges(liquid_fraction)
        Vg, Wl, V = calculate_velocities(liquid_fraction, pressure, D_g, cfg)

        new_salt = np.zeros((I,))
        new_gas = np.zeros((I,))
        new_pressure = np.zeros((I,))

        heat_flux = calculate_heat_flux(state_BCs, Wl, V, D_g)
        new_enthalpy = take_forward_euler_step(state.enthalpy, heat_flux, timestep, D_e)

        new_salt = salt_ghosts[1:-1] + timestep * (
            (1 / cfg.physical_params.lewis_salt)
            * np.matmul(
                D_e, liquid_fraction_edges * np.matmul(D_g, liquid_salinity_ghosts)
            )
            - np.matmul(D_e, upwind(salt_ghosts, V))
            - np.matmul(D_e, upwind(liquid_salinity_ghosts + C, Wl))
        )
        new_gas = gas_ghosts[1:-1] + timestep * (
            (chi / cfg.physical_params.lewis_gas)
            * np.matmul(
                D_e, liquid_fraction_edges * np.matmul(D_g, dissolved_gas_ghosts)
            )
            - np.matmul(D_e, upwind(gas_ghosts, V))
            - np.matmul(D_e, upwind(gas_fraction_ghosts, Vg))
            - np.matmul(D_e, upwind(chi * dissolved_gas_ghosts, Wl))
        )

        new_state = State(
            new_time,
            new_enthalpy,
            new_salt,
            new_gas,
        )
        new_state.calculate_enthalpy_method(cfg)

        new_liquid_fraction = new_state.liquid_fraction
        new_gas_fraction = new_state.gas_fraction
        new_permeability = calculate_absolute_permeability(new_liquid_fraction)

        new_pressure = solve_pressure_equation(
            gas_fraction, new_gas_fraction, new_permeability, timestep, D_e, D_g, cfg
        )
        new_state.pressure = new_pressure

        return new_state
