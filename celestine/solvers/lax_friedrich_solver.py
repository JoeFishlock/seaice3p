import numpy as np
from celestine.boundary_conditions import calculate_enthalpy_from_temp
from celestine.forcing import get_temperature_forcing
from celestine.enthalpy_method import (
    calculate_enthalpy_method,
    get_phase_masks,
)
from celestine.grids import (
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
from celestine.solvers.template import SolverTemplate


class LXFSolver(SolverTemplate):
    """Take timestep using LXF scheme

    Boundaries and solid regions still use upwind fluxes
    """

    def take_timestep(self, enthalpy, salt, gas, pressure, time, timestep):
        I = self.I
        chi = self.cfg.physical_params.expansion_coefficient
        C = self.cfg.physical_params.concentration_ratio
        D_g = self.D_g
        D_e = self.D_e

        new_time = time + timestep
        new_top_temperature = get_temperature_forcing(new_time, self.cfg)
        new_top_enthalpy = calculate_enthalpy_from_temp(
            0, 0, new_top_temperature, self.cfg
        )

        phase_masks = get_phase_masks(
            enthalpy,
            salt,
            gas,
            self.cfg,
        )
        (
            temperature,
            liquid_fraction,
            gas_fraction,
            _,
            liquid_salinity,
            dissolved_gas,
        ) = calculate_enthalpy_method(enthalpy, salt, gas, self.cfg, phase_masks)
        Vg, Wl, V = calculate_velocities(
            liquid_fraction, enthalpy, salt, gas, pressure, D_g, self.cfg
        )

        new_enthalpy = np.zeros((I + 2,))
        new_salt = np.zeros((I + 2,))
        new_gas = np.zeros((I + 2,))
        new_pressure = np.zeros((I + 2,))

        new_enthalpy[-1] = new_top_enthalpy
        new_enthalpy[0] = self.cfg.boundary_conditions_config.far_temp
        new_salt[-1] = 0
        new_salt[0] = 0
        new_gas[-1] = 0
        new_gas[0] = chi

        # Upwinding
        upwind_enthalpy_flux = (
            -(D_g @ temperature) + upwind(temperature, Wl) + upwind(enthalpy, V)
        )
        upwind_salt_flux = (
            -(1 / self.cfg.physical_params.lewis_salt)
            * (geometric(liquid_fraction) * (D_g @ liquid_salinity))
            + upwind(salt, V)
            + upwind(liquid_salinity + C, Wl)
        )
        upwind_gas_flux = (
            -(chi / self.cfg.physical_params.lewis_gas)
            * (geometric(liquid_fraction) * (D_g @ dissolved_gas))
            + upwind(gas, V)
            + upwind(gas_fraction, Vg)
            + upwind(chi * dissolved_gas, Wl)
        )

        # Lax Friedrich
        salt_no_flux = np.insert(salt[1:-1], 0, salt[0])
        salt_no_flux = np.append(salt_no_flux, salt[-1])

        gas_no_flux = np.insert(gas[1:-1], 0, gas[0])
        gas_no_flux = np.append(gas_no_flux, gas[-1])

        numerical_diffusivity = (self.step**2) / (2 * timestep)

        salt_frame_advection = V * average(salt)
        salt_liquid_advection = Wl * (average(liquid_salinity) + C)
        salt_diffusion = (
            -(1 / self.cfg.physical_params.lewis_salt)
            * geometric(liquid_fraction)
            * (D_g @ liquid_salinity)
        )
        salt_LF_diffusion = -numerical_diffusivity * (D_g @ salt_no_flux)
        LF_salt_flux = (
            salt_frame_advection
            + salt_liquid_advection
            + salt_diffusion
            + salt_LF_diffusion
        )

        gas_frame_advection = V * average(gas)
        gas_bubble_advection = average(gas_fraction) * Vg
        gas_liquid_advection = chi * Wl * average(dissolved_gas)
        gas_diffusion = (
            -(chi / self.cfg.physical_params.lewis_gas)
            * geometric(liquid_fraction)
            * (D_g @ dissolved_gas)
        )
        gas_LF_diffusion = -numerical_diffusivity * (D_g @ gas_no_flux)
        LF_gas_flux = (
            gas_frame_advection
            + gas_bubble_advection
            + gas_liquid_advection
            + gas_diffusion
            + gas_LF_diffusion
        )
        # this works to prevent salt diffusion in solid and for small gas bubbles
        # However for more general case condition for gas flux should perhaps be when
        # R_B = R_T
        is_solid = geometric(liquid_fraction) == 0
        # Must always upwind on the boundaries to avoid using incomplete information here
        is_solid[-1] = True
        is_solid[0] = True

        enthalpy_flux = upwind_enthalpy_flux
        salt_flux = np.where(is_solid, upwind_salt_flux, LF_salt_flux)
        gas_flux = np.where(is_solid, upwind_gas_flux, LF_gas_flux)

        new_enthalpy[1:-1] = enthalpy[1:-1] - timestep * (D_e @ enthalpy_flux)
        new_salt[1:-1] = salt[1:-1] - timestep * (D_e @ salt_flux)
        new_gas[1:-1] = gas[1:-1] - timestep * (D_e @ gas_flux)

        new_phase_masks = get_phase_masks(
            new_enthalpy,
            new_salt,
            new_gas,
            self.cfg,
        )
        (
            new_temperature,
            new_liquid_fraction,
            new_gas_fraction,
            _,
            new_liquid_salinity,
            new_dissolved_gas,
        ) = calculate_enthalpy_method(
            new_enthalpy, new_salt, new_gas, self.cfg, new_phase_masks
        )
        # toggle this to old liquid fraction to see if it makes any difference
        new_permeability = calculate_absolute_permeability(
            geometric(new_liquid_fraction)
        )

        new_pressure = solve_pressure_equation(
            gas_fraction,
            new_gas_fraction,
            new_permeability,
            timestep,
            D_e,
            D_g,
            self.cfg,
        )

        step, _, _, _ = initialise_grids(self.I)

        # Calculation for adaptive timestepping
        CFL_timesteps = (
            self.cfg.numerical_params.CFL_limit
            * step
            / np.where(np.abs(upwind(gas_fraction, Vg)) > 0, np.abs(Vg), 1e-10)
        )
        Courant_timesteps = self.cfg.numerical_params.Courant_limit * step**2
        CFL_min_timestep = np.min(CFL_timesteps)
        Courant_min_timestep = np.min(Courant_timesteps)
        min_timestep = min(CFL_min_timestep, Courant_min_timestep)

        return new_enthalpy, new_salt, new_gas, new_pressure, new_time, min_timestep
