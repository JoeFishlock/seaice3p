import numpy as np
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
from celestine.solvers.template import SolverTemplate


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


class ReducedSolver(SolverTemplate):
    """Take timestep using upwind scheme with liquid velocity calculation lagged."""

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

        Vg, Wl, V = calculate_velocities(state_BCs, D_g, cfg)
        Vg = prevent_gas_rise_into_saturated_cell(Vg, state_BCs)

        heat_flux = calculate_heat_flux(state_BCs, Wl, V, D_g)
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
