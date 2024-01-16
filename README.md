# Celestine #

Code for simulating gas content of sea ice in 1D using enthalpy method.

## Install ##

Currently use by downloading the source code and placing the `celestine/` directory next to your script.
Then import any functions you need as in the example `main.py`.
Requirements can be installed by running `pip install -r requirements.txt`.

## Usage ##

Save configurations for a simulation (either dimensional or non-dimensional but not a mixture) as yaml files.
This can be done by editing examples or by using classes within the dimensional_params and params modules.
Once you have a directory of configuration files the simulation for each can be run using `python -m celestine path_to_configuration_directory path_to_output_directory`.
The `--dimensional` flag should be added to this command if running dimensional parameter configurations.
The simulation will be run for each configuration and the data saved as a numpy archive with the same name as the simulation in the specified output directory.

## Documentation ##

found in the `docs/` directory

- `Changelog.md`
- `test_results.md` is breakdown of results of test simulations.
- `manual.pdf` is the sphinx generated documentation from docstrings.
Generate by running `make latexpdf` in the `docs/` directory and then copying the ouput in the `docs/build/` directory to `docs/manual.pdf`. 
- `numerical_method.pdf` is a written description of the numerical method used for each solver option.

## Tests ##

- Run `python -m tests.generate_tests` to add all test simulation yaml files to `test_data/` directory.
- Run `python -m tests.run_tests` to run all test configurations.
- Those that run or crash will be in the logs.
- Record simulations that crash.
- Others have run. Note tests that ran could still have garbage output.
- Collect this info in the `docs/test_results.md` file using the following template for each simulation.

TODO: add tests that use barrow forcing conditions and initial conditions

## Release checklist ##

- run tests and record results wth version number and time
- bump version number in celestine/__init__.py
- bump version number in sphinx documentation in docs/source/conf.py
- create docs by running make latexpdf in docs/ directory and put pdf from build directory into docs/
- update Changelog.md
- tag commit with version number
