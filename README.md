# seaice3p #

Code for simulating the seasonal evolution of a 1D layer of sea ice using an enthalpy method.
Inlcudes three-phase mushy-layer physics which can be used to simulate air bubbles or oil droplets in the ice.
See Joseph Fishlock's DPhil thesis for further details.

## Install ##
Seaice3p is available on PyPI and can be installed with pip via
```bash
pip install seaice3p
```

## Usage ##
Configurations to run simulations can be created via a Python script using the `DimensionalParams` object and are saved as yaml files.
Once you have a directory of configuration files the simulation for each can be run using `python -m seaice3p path_to_configuration_directory path_to_output_directory`.
The `--dimensional` flag should be added to this command if running dimensional parameter configurations.
The simulation will be run for each configuration and the data saved as a numpy archive with the same name as the simulation in the specified output directory.

## Documentation ##
Some incomplete API reference documentation built using `mkdocs gh-deploy` is available at
[documentation](https://joefishlock.github.io/seaice3p).

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
