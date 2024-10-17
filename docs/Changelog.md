# Changelog

## v0.22.0 (2024-10-17) ##

### Summary ###
To avoid delivering spurious advective heat flux in the ocean when using the RJW14 brine convection scheme we now use an exponential
to smoothly set the advective heat flux in the ocean to zero. This should improve the underestimate of ice growth compared to field
observations.

## v0.21.0 (2024-10-11) ##

### Summary ###
Update dependencies so pip installs xarray, metpy and netcdf4 needed for running simulations with forcing from ERA5 data.

To avoid numerical instability change the calculation of eddy diffusivity to turn on gradually as liquid fraction goes to 1 using an exponential.
Set the scale for the exponential once as 5e-3 which should mean the diffusivity remains unmodified for liquid fraction less than 0.9.
This seems to suppress instability and simulations can be run with the BDF solver for a year of ERA5 reanalysis forcing.

Change the calculation of the enthalpy method to account for different specific heat capacities of the solid and liquid phases.
This is important to predict the correct ice depth.

## v0.20.0 (2024-10-07) ##

### Summary ###
Add ERA5Forcing option for simulation configuration.
This reads data from a single ERA5 reanalysis netcdf data file at the location provided in configuration.
This file should onlu contain timeseries data for a single location (lat, lon location) and needs to contain the hourly variables:
2m air temperature,
2m dewpoint temperature,
downward longwave radiation.
downward shortwave radiation,
and surface pressure.
Optionally can also request to use dnow depth data from the data file.
In this case the surface energy balance boundary condition is modified assuming a quasi steady homogeneous conductive snow layer.
To read the netcdf file xarray and netCDF4 are added as dependencies as well as metpy to calculate specific humidity at 2m.

Seperate the ocean boundary conditions into a sperate ocean_forcing_config.
We have implemented three options:
FixedTempOceanForcing, which provides fixed ocean temperature boudnary conditions;
FixedHeatFluxOceanForcing, which provides a constant ocean heat flux;
and BRW09OceanForcing, which provides the bottom ocean temperature from the data measured at 2.4m during the 2009 Barrow field study.

Change from specifying the turbulent liquid thermal conductivity to simply specifying the eddy diffusivity.
In purely liquid regions this enhances the diffusion of heat, salt and dissolved gas.
Additionally added the option gas_bubble_eddy_diffusivity, when set to true this also adds eddy diffusion of the gas bubble phase.
This is useful when simulating oil droplets instead of gas bubbles which are much less buoyant and should be mixed due to turbulence in the liquid.

Added the gas_viscosity parameter.
By default this is zero and we regain the terminal rise velocity calculation for a free slip sphere.
However, when a non-zero value is supplied the Hadamaard-Rybczinski equation is used.

Fixed a bug when implementing the brine convection sink so that now we may couple brine convection to the gas / oil droplet phase by setting
the options couple_bubble_to_horizontal_flow and couple_bubble_to_vertical_flow to true.
This is useful for simulating oil droplets or bubbles which have become trapped and are not migrating under their buoyancy.

Added the dates property to the results class which can be calcuated for simulations forced useing ERA5Forcing as we specify a start date for these.
Gives a list of datetimes of the data points of the simulation.
Added methods to calculate meltpond onset time and surface heat fluxes of a simulation to the results class.

## v0.19.0 (2024-09-30) ##

### Summary ###
Change to use oilrad v0.7.0 for radiative transfer calculation.
This allows us to use an optimized faster radiation solve if we set fast_solve=True.
Given a cutoff wavelength you should choose based on the grid resolution for wavelengths above this
approximate the radiation as entirely absorbed in the first grid cell.

Add turbulent liquid thermal conductivity parameter to represent enhanced heat transport in liquid regions (ocean and meltpond).
Also add the option to choose the solver used by scipy.integrate.solve_IVP.
An implicit method such as the Radau or LSODA option is best used when an enhanced turbulent conductivity is added to avoid very small timesteps.

Add option to load simulation initial condition as the final state of an existing saved simulation.

Make gas fraction boundary condition extend the bottom value of the domain into the ocean.
This is useful for simulating an oil configuration where oil will be able to rise into the domain.

Add plot module to visualise saved simulation data from the command line.

Add the initial_oil_free_ice_depth parameter to the oil simulation initial conditions config to specify the initial oil mass ratio profile as a step function with no oil in the upper portion of the domain.
This is to simulate the release of oil below the ice at some time before the simulation starts.

## v0.18.0 (2024-09-26) ##

### Summary ###
Change to use oilrad v0.6.0 for radiative transfer calculation.
Fix bug where non-dimensional grid was passed to oilrad causing incorrect internal melting.
Fix implementation of surface energy balance to estimate surface temprature correctly.
Create the load_simulation function to return a results class useful for plotting simulations.
Add a forcing option for a Robin boundary condition.
Fix bug in non-dimensionalisation of DISEQ model so that this can now be run.

## v0.17.0 (2024-09-21) ##

### Summary ###
Change to use oilrad v0.5.0 for radiative transfer calculation.
This just takes liquid fraction within the entire domain and so solves a depth dependent radiative transfer model with meltpond and ocean regions present.
Once a meltpond forms on the surface of the ice the surface SW penetration fraction is set to 1 as there can be no SSL when a meltpond has formed.

## v0.16.0 (2024-09-19) ##

### Summary ###
Change to use oilrad v0.4.0 for radiative transfer calculation.
This now integrates accross all shortwave wavelengths so we no longer need to have the shortwave in the surface boundary condition.
The SW_penetration_fraction dictates how much radiation passes through the initial surface scattering layer to the solver.

## v0.15.0 (2024-09-18) ##

### Summary ###
Renamed the project from the working name celestine to seaice3p.
Updated to use python 3.12.
Added verbosity option to command line interface -v.
Removed logging to files.
Major refactoring.

Removed redundant enthalpy method and solver classes.
Simulation state is stored in the State object which keeps the prime variables.
Running the enthalpy method on this onbect produces a StateFull object with the enthalpy method variables.
Applying the boundary conditions to this object produces the StateBCs object used in the solver.
Broken up to configuration of the simulation to be handled by different objects.
This should mean only necessary parameters for the type of simulation being run need to be given.
Use pyserde to serialize these configuration objects.
Added a load module to read in data from simulations.

Implemented a disequilibrium model for gas dynamics with a finite nucleation rate.

Added convenience function to adapt gas dynamics to simulate oil droplets.
The buoyancy parameter is now calculated as the difference in fluid and oil/gas density.
When simulating oil droplets the top velocity is changed to prevent oil escaping the top surface.

Implemented a radiative forcing configuration which uses a surface energy balance to calculate
the appropriate temperature boundary condition.
Also use the radiative transfer model `oilrad` (v0.3.0) to calculate internal radiative heating due to 
shortwave absorption of ice and oil droplets.
Created an initial condition to investigate melting a layer of ice under radiative and turbulent
surface fluxes.

## v0.14.0 (2024-04-23) ##

### Summary ###
No new physics in this version just changed the structure so that tests are run with pytest.
Example script is now a module `celestine.example`.
Scripts that plotted gas velocity and brine drainage parameterisation quantities were useful so have been moved to
a diagnostics module.

### Tests ###
- Run all tests with pytest.
- Run tests that aren't slow with `pytest -m "not slow"`.

### Docs ###
- Update installation instructions in the README.

## v0.13.0 (2024-04-22) ##

### Summary ###
To investigate the effect of gas nucleation rate without changing the equilibrium model significantly add parameters
that control tolerable supersaturation and ocean saturation state. This allows investigation of a case with less gas
exsolution.

Package is also now pip installable.

### Added ###
- Tolerable supersaturation parameter. Used by ReducedEnthalpyMethod.

## Changed ##
- Barrow initial condition initialises ocean with `far_gas_sat` so that ocean can be subsaturated with air.
- Made poetry a package managed by poetry with pyproject.toml so that it can be installed easily with pip.

## Removed ##
- Remove requirements.txt as not needed now dependencies are in pyproject.toml.

## v0.12.0 (2024-01-30) ##

### Summary ###
Add the option to run simulations with phase averaged thermal conductivity to better match ice depth for real sea
ice growth. We also add the functionality to run a single simulation config or a directory of simulation configs,
which can be dimensional or not, from the command line using the `python -m celestine` command.

### Added ###
- Add parameters to the dimensional params configuration class so that a simulation configuration can be generated
without supplying any additional objects.
- Add __main__.py file which uses argparse to run simulation configurations from the command line.
- Option to calculate conductive heat transfer using phase averaged thermal conductivities.

## Changed ##
- Remove data_path as a configuration option and change solver methods to specify where they will save output.
- Flux module calculation to calculate phase average thermal conductivities.

## Removed ##
- Remove tests/run_tests.py as tests can now be run from the command line using `python -m celestine test_data`.

### Docs ###
- Update test procedure in README.

## v0.11.0 (2024-01-15) ##

### Summary ###
We have added options to the simulation to use a brine convection parameterisation (Rees Jones 2014).
This desalinates the ice and brings in saturated ocean water, assuming continuity of velocity at the ice interface.
This seems to work relatively well to desalinate the ice for the barrow simulation if we start with ocean bulk salinity.
This however also changes the heat balance so to get the ice growth correct we need to use a thermal conductivity value
that is an average of ice and water. Going forward we could just implement the full heat conduction term for each phase.
To further improve the barrow simulation we use thermistor data for the ocean to force the bottom of the domain.
We also give the option to use thermistor data at the ice snow interface to better match temperature evolution of the ice.
This is better than the air temperature as we do not simulate the insulating layer of snow.

### Added ###
- Functionality to calculate the liquid velocity associated with brine convection using Rees Jones
2014 parameterisation by turning the brine_convection_parameterisation to True in the simulation configuration.
The parameterisation should advect tracers with the broad upward liquid flow and also remove salt, heat and bulk gas
via a sink term that appears as the downward brine channel flow. There are two more true/false flags that
decide wether to couple bubble motion to the vertical flow that should move bubbles upward and to the horizontal flow
which is responsible for transporting bubbles to brine channels where they would be expelled and so this appears in
the sink term.
- Added configuration parameters needed for the brine convection parameterisation. The critical Rayleigh number,
the convection strength tuning parameter and then the dimensional haline contraction coefficient and reference permeability.
The two tuning parameters are given default values from the Rees Jones 2014 paper but later work (Thomas 2022) suggests
using lower values of these will work better to desalinate the ice.
- test_brine_drainage.py This script is useful as it generates some plots illustrating the functions
used to calculate the ice depth, rayleigh number and convecting liquid velocity.
This can be used to visually confirm the parameterisation is working as expected and the templates for
plotting these quantities may come in handy.
- drainage_test.py This script runs a simulation with the brine drainage parameterisation turned on.
- celestine/brine_drainage.py This module calculates the quantities needed for the Rees Jones 2014
brine convection parameterisation and provides the parameterised darcy liquid velocity to the rest of
the simulation.
- celestine/brine_channel_sink_terms.py This module implements the loss of heat, salt and bulk gas
through the downward brine channel flow in the Rees Jones 2014 convective parameterisation.
This provides the terms in the conservation equations that loose heat, salt and bulk gas to the ocean.
- Added the option to choose the thermistor temperature data used to force the top of the simulation for the barrow simulation.
This is important as we don't simulate a snow layer so we can choose via the new option Barrow_top_temperature_data_choice
in the configuration if we want to use temperature data recorded at the air interface, bottom snow or top of ice.
- Added option in the barrow configuration to choose the bulk gas content of the initial ice cover.

## Changed ##
- Make the barrow simulation configuration use recorded ocean temperature to force the bottom of the domain.

### Docs ###
- Add the modules brine_drainage and brine_channel_sink_terms to the documentation index.

### Bugs ###
- The brine convection parameterisation seems to work but the option to couple bubbles to the horizontal flow
and hence remove free gas phase via brine channels does not work as it seems some quantity is calculated on the wrong
grid. This option currently just breaks the simulation if set to True.

## v0.10.0 (2023-11-24) ##

### Summary ###
To calculation of gas velocity we add options to use a different fit for wall drag enhancement
function taken from a paper by Haberman. We also add the option to use a critical liquid
velocity percolation threshold to cut off gas motion. We add some plots comparing different
gas rise parameterisations.

### Added ###
- Alternative fit for wall drag enhancement as a function of scaled bubble radius taken from
a paper by Haberman.
- parameters in config and non dimensional config to choose the type of wall drag funciton used.
- plot of different wall drag enhancement functions and bubble rise velocities against liquid
fraction for different bubble distributions and drag laws.
- Add critical liquid fraction porosity cutoff of 2.4% from Maus paper.
- Add options to enable this cutoff, leave the default behaviour the same.
- Gas velocity plots comparing different bubble terminal rise velocities to Moreau 2014 paper.

## Changed ##
- Values for pore throat radius and scaling taken from Maus paper for gas velocity plots.
Before we were wrongly using diameter instead of radius.
- Change default value of dynamic liquid viscosity used in dimensional configuration to
match the value of kinematic viscosity used in Moreau 2014 paper.

## v0.9.0 (2023-11-12) ##

### Summary ###
This version adds the funcitonality to calculate the gas Darcy flux using the interstitial
terminal rise velocity of a bubble averaged over a power law bubble size distribution.
The default behaviour remains to use a single bubble size but parameters now exist for
the power law case. The velocities module has also been refactored and in anticipation
of parameterising the liquid flow instead of direct calculation the lagged upwind solver
and funcitons for solving the pressure ODE are removed here.

### Added ###
- script called plot_gas_velocity.py to plot different versions of the calculated gas
interstitial velocity against liquid fraction.
- Functions in the velocities module to calculate the gas interstitial velocity
averaged over a power law distribution of bubble sizes.
- Parameters needed (dimesnional and non dimensional) to select either a single bubble size
or power law distribution case. In the power law case added the maximum and minimum bubble
sizes and the power law slope as parameters.

## Changed ##
- Refactor calculation of gas interstitial velocity to make it possible to add options
to calculate this with a monodispersed bubble size distribution or a power law distribution.
- The definition of the non dimensional buoyancy parameter B is changed to use the pore
length scale as this doesn't change as we integrate over bubble size distributions.

### Removed ###
- For simplicity we remove the functions which calculate liquid Darcy velocity from
solving an ODE for the pressure at each timestep. These are not necessary for the
reduced model approximation.
- Remove the lagged upwind solver which was the only one to attempt to use the pressure
solve.

### Docs ###
- Equation for calculation of gas bubble interstitial velocity is updated in the numerical
method documentation.

### Tests ###
- Remove test cases that use the now removed lagged upwind solver.


## v0.8.0 (2023-05-23) ##

### Summary ###

This code can now generate a simulation configuration from dimensional parameter inputs.
It can also convert the output to dimensional units for plots.
It can also now run simulations with the "barrow_2009" forcing and initial conditions option
which uses surface temperature data from the Barrow field station in 2009 to compare our
simulation data to the field data of Zhou and Tison.

### Added ###

- Dimensional parameters module to handle input of dimensional parameters and converting
between non dimensional and dimensional variables.
- Example script to run a simulation with Barrow 2009 Jan-Jun configuration called `barrow.py`.
- Barrow field station temperature data and metadata in `celestine/forcing_data/`. Must be read
in to use "barrow_2009" temperature forcing option.
- Initial conditions module so we can use different initial conditions chosen in configuration.
- Method in Scales class to convert bulk air content into mircro moles of Argon per Liter of ice,
under some assumptions that mass ratio of Argon in air is the same as in the atmosphere. This lets
us compare to the field data for Argon of Zhou and Tison.

### Docs ###

- Document explaining numerical methods used. 
- README containing install instructions, how to run the tests, a breakdown of the documentation
and checklist for creating a new release.

## v0.7.0 (2023-05-19) ##

### Summary ###

This code is now capable of solving the reduced model configuration for the same forcing
and boundary conditions as the full model. This is a set of approximations where the gas
fraction is neglected in the enthalpy method and so no liquid flow is generated by gas
motion, so we do not need to solve for the liquid pressure. The reduced model can be solved
using a forward euler upwind scheme or using RK23 with scipy solve_ivp. As this includes
adaptive timestepping this works well for simulations with high gas buoyancy. It is 
important to note that as gas fraction is decoupled from solid and liquid fraction in
the reduced model we must impose that gas cannot enter a cell which already contains high
enough gas fraction to saturate the pore space.

### Added ###

- Reduced model classes for the phase boundaries, enthalpy method and solver. This is chosen
with the solver choice "RED" in the simulation configuration.
- Solver class to solve the reduced model using the `scipy.integrate.solve_ivp` function
with the RK23 method. This solver option is "SCI" in the simulation configuration.

### Docs ###

- Add docs pages for the reduced model solver and the reduced model solver that uses
`scipy.integrate.solve_ivp`.

### Tests ###

- Include configurations with the two new solvers in the test cases.

## v0.6.0 (2023-05-19) ##

## Summary ##

Can run only the lagged upwind solver (LU) for the full enthalpy method with constant or
yearly surface temperature forcing with given initial state. A lot of broken or redundant
code was removed from v0.5.0 and refactored to make it easier to extend adding reduced model
as another enthalpy method and new solvers and boundary conditions.

### Added ###

- Helper function for grids to add ghost cells to a quantity on cell centers. This is very
useful for applying boundary conditions.
- Module called flux to calculate fluxes for heat, salt and gas. This should make
implementing new solvers easier and more reliable.
- Function for lagged upwind solver to take forward euler timestep given fluxes.
- State module for working with solution state at each timestep. Contains the State
class for working with the solution on cell centers at each timestep and running the
appropriate enthalpy method. StateBCs is responsible for adding the boundary conditions.
Solution is responsible for storing the timesteps to be saved.
- Class interface for enthalpy method so we can implement different versions. The
enthalpy method is picked from the solver choice in the configuration.
- Class interface for phase boundary calculation so we can implement difference versions.

### Changed ###

- Refactor solver template and enthalpy method to use a state class. 
This contains all the variables needed at a timestep so makes writing solvers more concise.
- Solver template uses solution class to store variables to be saved in a numpy array of
fixed dimensions. This has better performance than appending to an array each time we
want to save new data.
- Simplify boundary conditions module to just add a fixed condition to each variable we may
need on the ghost grid. Then the StateBCs class handles adding these to the state at each
timestep.
- Before boundary conditions were calculated by inverting the enthalpy method. Now we just
impose appropriate values of temperature, dissolved gas etc... I have chosen to extrapolate
the liquid fraction to the ghost cells as being equal to the top and bottom cell centers.
- Solution output of simulation is now given on the cell centers but can be easily extended
to the ghost cells by using the StateBCs class.
- Initial solution is now uniform profile of enthalpy, bulk salt and bulk gas given by their
given values at the bottom of the domain.
- Lagged Upwind solver uses new template solver interface.
- Velocities module now takes stateBCs object.

### Docs ###

- Auto generate docstring documentation for each module using sphinx as `docs/manual.pdf`.

### Tests ###

- Put tests in their own package `tests/`.
- Run different bubble sizes and constant/yearly forcing for only lagged upwind solver as
test case. Record if the simulation runs or crashes.

### Removed ###

- Implicit lax friedrich solver (LXFImplicit).
- Lax Friedrich solver (LXF).
- Other partially written solvers that I wasn't using.
- Adaptive timestepping option. This wasn't really being utilised and was making the code
more opaque.
- Checks on initial timestep and grid size. Now just log a warning if Courant number for
thermal diffusion exceeds 0.5 in the lagged upwind solver as this treats the diffusive term
explicitly.
- Code to plot full enthalpy method phase space diagram from `celestine.phase_boundaries.py`.
- Code to plot enthalpy method quantities from `celestine.enthalpy_method.py`.
- Scripts to plot benchmark case for comparing the three solvers in v0.5.0.

## v0.5.0 (2023-04-27) ##

## Summary ##

Runs full enthalpy method with three different solvers:
- LU (forward euler explicit upwind scheme, calculate velocity from previous timestep)
- LXF (forward euler explicit lax-friedrich scheme)
- LXFImplicit (same as LXF but calculate heat diffusion implictly for better resolution)

Contains benchmark scripts to compare these solvers during yearly temperature forcing run.

Contains code in enthalpy_method and phase_boundaries to plot phase space diagrams.

Contained artificial cut off in liquid darcy velocity calculation and all of the solvers
suffer the same instability during melting part of yearly cycle.

### Added ###

- Options to forcing configuration to control sinusoidal yearly temperature forcing.
- LXFImplicit solver option which is the same as lax friedrich solver but uses backwards
Euler for temperature diffusion terms. This relaxes timestep constraint so that grid can 
be refined to investigate numerical diffusion introduced by this method.
- Scripts to generate benchmark configuration, run the simulations and plot the data.
These scripts compare the LU, LXF and LXFImplicit solvers for a yearly temperature forcing
run with micro bubbles and plot the profiles from the different solvers on the same axis to
compare.

### Changed ###

- Example simulation in main.py now plots solid fraction and temperature.
- Adaptive timestepping now controlled by flag in nuerical params object.
Default is false so solve will just proceed with initial timestep.
- Calculate liquid Darcy velocity with permeability as the cube of liquid fraction.
This reverts a change made to debug instability which I forgot about.
- Pressure solver regularisation increased from 1e-15 to 1e-4 so it can have more of
an effect.

### Tests ###

- Add LXFImplicit solver to manual testing cases.

## v0.4.0 (2023-04-18) ##

### Added ###

- Lagged upwind solver as an option to use.

### Changed ###

- Refactored solvers to use a template solver which handles common methods.
Implemented solvers should inherit and overwrite the relevant methods.

## Docs ##

- Specified a testing procedure to run for each release.

### Tests ###

- Added script to generate test simulation config yaml files in `test_data/`.
- Added script to run test simulation config and log if any crash.
- This information can be recorded manually in the template file in `docs/`.

## v0.3.0 (2023-04-17) ##

### Added ###

- Logging. Logs are generated for the simulation in main stating the version
number and duration of the simulation stored in the logs/ directory.

### Changed ###

- Split the Params class into multiple different classes. These are combined
in the new Config class. 
- Code refactored to use the Config class (only the lax friedrich solver not
the others).
- Config object can be saved to and loaded from a yaml file not json.

### Docs ###

- Added release checklist.

### Removed ###

- Removed dependency on tqdm for progress bar.

## v0.2.0 (2023-04-14) ##

### Docs ###

- add requirements in requirements.txt file

### Added ###

- lax friedrich solver (still upwind enthalpy, boundary values and solid regions)

## Removed ##

- remove dependency on scienceplots

### Changed ###

- change yearly temperature forcing period to be 4 units of time
- function to compute cell edge values as arithmetic mean of centers
- main function just run one yearly simulation with small bubbles and plot gas fraction and bulk salinity for debugging

## v0.1.0 (2023-04-13)

- Simulate bubbles in a mushy layer in 1D using the enthalpy method with given surface forcing.
- Phase fractions of solid, liquid and gas must sum to one and so bubble nucleation and motion drives a liquid flow.
- Implemented a variety of implicit and semi-implicit solvers as calculating liquid velocity from pressure solver requires knowlege of gas fraction time derivative.
- Backward euler approach uses scipy root findding (krylov solver) to solve non linear system of residuals.
- Iterative solver performs a fixed point iteration for each timestep until residuals are suitably lows.
- Lagged solver assumes velocity calculated in between each timestep, this will introduce some error in the flow calculated.

# Problems

- All of these methods seem to suffer instabilities when gas accumulates below an impermeable eutectic layer which melts as surface temperature forcing warms.
