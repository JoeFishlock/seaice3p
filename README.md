# seaice3p #

Code for simulating the seasonal evolution of a 1D layer of sea ice using an enthalpy method.
Optionally can simulate the air fraction within the ice or the motion of oil droplets.

## Install ##

Install via pip (v0.13.0 and above):
`pip install seaice3p`

To install a specific version run `pip install git+file:///ABSOLUTE/PATH/TO/LOCAL/GIT/REPO@vX.X.X`.

## Usage ##

Save configurations for a simulation (either dimensional or non-dimensional but not a mixture) as yaml files.
This can be done by editing examples or by using classes within the dimensional_params and params modules.
Once you have a directory of configuration files the simulation for each can be run using `python -m seaice3p path_to_configuration_directory path_to_output_directory`.
The `--dimensional` flag should be added to this command if running dimensional parameter configurations.
The simulation will be run for each configuration and the data saved as a numpy archive with the same name as the simulation in the specified output directory.
Example script that generates, runs and plots a simulation can be run with `python -m seaice3p.example`.

## Documentation ##

Files to generate documentation using mkdocs found in the `docs/` directory as well as `Changelog.md`.

## Tests ##

Run `pytest` to run all tests.
Note this may take some time so you can also run `pytest -m "not slow"`.
To speed this up run in parallel using `pytest-xdist` with the extra options `pytest -n auto --dist worksteal`.

## Release checklist ##

- run tests.
- bump version number in seaice3p/__init__.py and pyproject.toml
- run `mkdocs build` to generate documentation and deploy from main with `mkdocs gh-deploy`.
- update Changelog.md
- tag commit with version number
