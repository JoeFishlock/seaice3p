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
from celestine.velocities import calculate_velocities, calculate_absolute_permeability
from celestine.solvers.template import SolverTemplate


class LaggedUpwindSolver(SolverTemplate):
    """Take timestep using upwind scheme with liquid velocity calculation lagged."""

    def take_timestep(self, enthalpy, salt, gas, pressure, time, timestep):
        I = self.I
        chi = self.cfg.physical_params.expansion_coefficient
        C = self.cfg.physical_params.concentration_ratio
        cfg = self.cfg
        D_g = self.D_g
        D_e = self.D_e

        new_time = time + timestep
        top_temperature = get_temperature_forcing(time, cfg)
        new_top_temperature = get_temperature_forcing(new_time, cfg)
        top_enthalpy = calculate_enthalpy_from_temp(0, 0, top_temperature, cfg)
        new_top_enthalpy = calculate_enthalpy_from_temp(0, 0, new_top_temperature, cfg)

        phase_masks = get_phase_masks(
            enthalpy,
            salt,
            gas,
            cfg,
        )
        (
            temperature,
            liquid_fraction,
            gas_fraction,
            _,
            liquid_salinity,
            dissolved_gas,
        ) = calculate_enthalpy_method(enthalpy, salt, gas, cfg, phase_masks)
        Vg, Wl, V = calculate_velocities(
            liquid_fraction, enthalpy, salt, gas, pressure, D_g, cfg
        )

        new_enthalpy = np.zeros((I + 2,))
        new_salt = np.zeros((I + 2,))
        new_gas = np.zeros((I + 2,))
        new_pressure = np.zeros((I + 2,))

        new_enthalpy[-1] = new_top_enthalpy
        new_enthalpy[0] = cfg.far_temp
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
            (1 / cfg.lewis_salt)
            * np.matmul(
                D_e, geometric(liquid_fraction) * np.matmul(D_g, liquid_salinity)
            )
            - np.matmul(D_e, upwind(salt, V))
            - np.matmul(D_e, upwind(liquid_salinity + C, Wl))
        )
        new_gas[1:-1] = gas[1:-1] + timestep * (
            (chi / cfg.lewis_gas)
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

        pressure_forcing = np.zeros((I + 2,))
        pressure_forcing[1:-1] = (1 / timestep) * (
            new_gas_fraction[1:-1] - gas_fraction[1:-1]
        ) + np.matmul(D_e, upwind(new_gas_fraction, V))
        pressure_forcing[0] = 0
        pressure_forcing[-1] = 0
        pressure_matrix = np.zeros((I + 2, I + 2))
        perm_matrix = np.zeros((I + 1, I + 1))
        np.fill_diagonal(perm_matrix, new_permeability + 1e-15)
        pressure_matrix[1:-1, :] = np.matmul(D_e, np.matmul(-perm_matrix, D_g))
        pressure_matrix[0, 0] = 1
        pressure_matrix[-1, -1] = 1
        pressure_matrix[-1, -2] = -1
        new_pressure = np.linalg.solve(pressure_matrix, pressure_forcing)

        step, _, _, _ = initialise_grids(cfg.I)

        CFL_timesteps = (
            cfg.CFL_limit
            * step
            / np.where(np.abs(upwind(gas_fraction, Vg)) > 0, np.abs(Vg), 1e-10)
        )
        Courant_timesteps = cfg.Courant_limit * step**2
        CFL_min_timestep = np.min(CFL_timesteps)
        Courant_min_timestep = np.min(Courant_timesteps)
        min_timestep = min(CFL_min_timestep, Courant_min_timestep)

        return new_enthalpy, new_salt, new_gas, new_pressure, new_time, min_timestep
