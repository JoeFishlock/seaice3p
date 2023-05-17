import numpy as np
from celestine.boundary_conditions import calculate_enthalpy_from_temp
from celestine.forcing import get_temperature_forcing
from celestine.enthalpy_method import (
    calculate_enthalpy_method,
    get_phase_masks,
    calculate_temperature,
    calculate_liquid_fraction,
    calculate_gas_fraction,
)
from celestine.grids import (
    get_difference_matrix,
    get_number_of_timesteps,
    upwind,
    geometric,
    initialise_grids,
    average,
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

        top_enthalpy = calculate_enthalpy_from_temp(0, 0, top_temperature, cfg)
        new_top_enthalpy = calculate_enthalpy_from_temp(0, 0, new_top_temperature, cfg)

        # calculate temperature, salinity etc for state on grid centers
        state.calculate_enthalpy_method(cfg)
        liquid_fraction_centers = state.liquid_fraction
        enthalpy_centers = state.enthalpy
        salt_centers = state.salt
        gas_centers = state.gas
        pressure_centers = state.pressure

        Vg, Wl, V = calculate_velocities(
            liquid_fraction, enthalpy, salt, gas, pressure, D_g, cfg
        )

        new_enthalpy = np.zeros((I + 2,))
        new_salt = np.zeros((I + 2,))
        new_gas = np.zeros((I + 2,))
        new_pressure = np.zeros((I + 2,))

        new_enthalpy[-1] = new_top_enthalpy
        new_enthalpy[0] = cfg.boundary_conditions_config.far_temp
        new_salt[-1] = 0
        new_salt[0] = 0
        new_gas[-1] = 0
        new_gas[0] = chi

        new_enthalpy[1:-1] = enthalpy[1:-1] + timestep * (
            np.matmul(D_e, np.matmul(D_g, temperature))
            - np.matmul(D_e, upwind(temperature, Wl))
            - np.matmul(D_e, upwind(enthalpy, V))
        )
        new_salt[1:-1] = salt[1:-1] + timestep * (
            (1 / cfg.physical_params.lewis_salt)
            * np.matmul(
                D_e, geometric(liquid_fraction) * np.matmul(D_g, liquid_salinity)
            )
            - np.matmul(D_e, upwind(salt, V))
            - np.matmul(D_e, upwind(liquid_salinity + C, Wl))
        )
        new_gas[1:-1] = gas[1:-1] + timestep * (
            (chi / cfg.physical_params.lewis_gas)
            * np.matmul(D_e, geometric(liquid_fraction) * np.matmul(D_g, dissolved_gas))
            - np.matmul(D_e, upwind(gas, V))
            - np.matmul(D_e, upwind(gas_fraction, Vg))
            - np.matmul(D_e, upwind(chi * dissolved_gas, Wl))
        )

        new_phase_masks = get_phase_masks(
            new_enthalpy,
            new_salt,
            new_gas,
            cfg,
        )
        (
            new_temperature,
            new_liquid_fraction,
            new_gas_fraction,
            _,
            new_liquid_salinity,
            new_dissolved_gas,
        ) = calculate_enthalpy_method(
            new_enthalpy, new_salt, new_gas, cfg, new_phase_masks
        )
        new_permeability = calculate_absolute_permeability(
            geometric(new_liquid_fraction)
        )

        new_pressure = solve_pressure_equation(
            gas_fraction, new_gas_fraction, new_permeability, timestep, D_e, D_g, cfg
        )

        step, _, _, _ = initialise_grids(self.I)

        CFL_timesteps = (
            cfg.numerical_params.CFL_limit
            * step
            / np.where(np.abs(upwind(gas_fraction, Vg)) > 0, np.abs(Vg), 1e-10)
        )
        Courant_timesteps = cfg.numerical_params.Courant_limit * step**2
        CFL_min_timestep = np.min(CFL_timesteps)
        Courant_min_timestep = np.min(Courant_timesteps)
        min_timestep = min(CFL_min_timestep, Courant_min_timestep)

        return new_enthalpy, new_salt, new_gas, new_pressure, new_time, min_timestep
