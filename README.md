# seaice3p #

Code for simulating gas content of sea ice in 1D using enthalpy method.

## Install ##

Install via pip (v0.13.0 and above).
To install a specific version run `pip install git+file:///ABSOLUTE/PATH/TO/LOCAL/GIT/REPO@vX.X.X`.
Alternatively use poetry to build a wheel for a specific version and pip install this.
Note depends on oilrad (v0.7.0).

## Usage ##

Save configurations for a simulation (either dimensional or non-dimensional but not a mixture) as yaml files.
This can be done by editing examples or by using classes within the dimensional_params and params modules.
Once you have a directory of configuration files the simulation for each can be run using `python -m seaice3p path_to_configuration_directory path_to_output_directory`.
The `--dimensional` flag should be added to this command if running dimensional parameter configurations.
The simulation will be run for each configuration and the data saved as a numpy archive with the same name as the simulation in the specified output directory.
Example script that generates, runs and plots a simulation can be run with `python -m seaice3p.example`.

## Documentation ##

found in the `docs/` directory

- `Changelog.md`
- `manual.pdf` is the sphinx generated documentation from docstrings.
Generate by running `make latexpdf` in the `docs/` directory and then copying the ouput in the `docs/build/` directory to `docs/manual.pdf`. 
- `numerical_method.pdf` is a written description of the numerical method used for each solver option.

## Tests ##

Run `pytest` to run all tests.
Note this may take some time so you can also run `pytest -m "not slow"`.
To speed this up run in parallel using `pytest-xdist` with the extra options `pytest -n auto --dist worksteal`.

## Release checklist ##

- run tests.
- bump version number in seaice3p/__init__.py
- bump version number in sphinx documentation in docs/source/conf.py
- bump version number in pyproject.toml
- run `sphinx-apidoc -o docs/source seaice3p` to generate source files for documentation.
- create docs by running make latexpdf in docs/ directory and put pdf from build directory into docs/
- update Changelog.md
- tag commit with version number
