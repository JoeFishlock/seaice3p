import numpy as np
from celestine.forcing import get_temperature_forcing
from celestine.grids import (
    upwind,
    add_ghost_cells,
    centers_to_edges,
)
from celestine.velocities import (
    calculate_velocities,
    calculate_absolute_permeability,
    solve_pressure_equation,
)
from celestine.solvers.template import SolverTemplate, State


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

        top_temperature = get_temperature_forcing(time, cfg)
        new_top_temperature = get_temperature_forcing(new_time, cfg)
        far_temp = cfg.boundary_conditions_config.far_temp
        far_gas_sat = cfg.boundary_conditions_config.far_gas_sat
        far_bulk_salt = cfg.boundary_conditions_config.far_bulk_salinity

        state.top_temperature = top_temperature

        # calculate temperature, salinity etc for state on grid centers
        state.calculate_enthalpy_method(cfg)

        enthalpy = state.enthalpy
        salt = state.salt
        gas = state.gas
        pressure = state.pressure
        liquid_fraction = state.liquid_fraction
        temperature = state.temperature
        liquid_salinity = state.liquid_salinity
        gas_fraction = state.gas_fraction
        dissolved_gas = state.dissolved_gas

        dissolved_gas_ghosts = add_ghost_cells(dissolved_gas, bottom=far_gas_sat, top=1)
        gas_fraction_ghosts = add_ghost_cells(gas_fraction, bottom=0, top=0)
        gas_ghosts = add_ghost_cells(gas, bottom=chi * far_gas_sat, top=chi)

        liquid_salinity_ghosts = add_ghost_cells(
            liquid_salinity, bottom=far_bulk_salt, top=liquid_salinity[-1]
        )

        temperature_ghosts = add_ghost_cells(
            temperature, bottom=far_temp, top=top_temperature
        )
        enthalpy_ghosts = add_ghost_cells(enthalpy, bottom=far_temp, top=enthalpy[-1])
        salt_ghosts = add_ghost_cells(salt, bottom=far_bulk_salt, top=salt[-1])

        liquid_fraction_edges = centers_to_edges(liquid_fraction)

        Vg, Wl, V = calculate_velocities(liquid_fraction, pressure, D_g, cfg)

        new_enthalpy = np.zeros((I,))
        new_salt = np.zeros((I,))
        new_gas = np.zeros((I,))
        new_pressure = np.zeros((I,))

        new_enthalpy = enthalpy + timestep * (
            np.matmul(D_e, np.matmul(D_g, temperature_ghosts))
            - np.matmul(D_e, upwind(temperature_ghosts, Wl))
            - np.matmul(D_e, upwind(enthalpy_ghosts, V))
        )
        new_salt = salt + timestep * (
            (1 / cfg.physical_params.lewis_salt)
            * np.matmul(
                D_e, liquid_fraction_edges * np.matmul(D_g, liquid_salinity_ghosts)
            )
            - np.matmul(D_e, upwind(salt_ghosts, V))
            - np.matmul(D_e, upwind(liquid_salinity_ghosts + C, Wl))
        )
        new_gas = gas + timestep * (
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
            top_temperature=new_top_temperature,
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
