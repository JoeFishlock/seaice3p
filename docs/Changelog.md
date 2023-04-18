# Changelog

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
