import argparse
from pathlib import Path
from . import __version__
from .printing import get_printer
from .params import DimensionalParams, Config, get_config
from .run_simulation import run_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configuration_directory",
        help="""load simulation configurations contained within this directory.\n
        This is defined as all files with .yml or .yaml extension.""",
    )
    parser.add_argument(
        "output_directory",
        help="save simulation output to this directory",
        nargs="?",
    )
    parser.add_argument(
        "-d",
        "--dimensional",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="This flag makes the program expect configurations with dimensional parameters",
    )
    parser.add_argument(
        "-s",
        "--single",
        help="""Use this option to give the file for a single configuration to run
        in the configuration directory instead of running all of the yaml files.""",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args()
    optprint = get_printer(args.verbose)

    is_dimensional_configuration = args.dimensional
    configuration_directory_path = Path(args.configuration_directory)

    if args.output_directory is None:
        output_directory_path = configuration_directory_path
    else:
        output_directory_path = Path(args.output_directory)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    optprint(f"Running seaice3p version: {__version__}")
    optprint(f"Save simulation output to: {output_directory_path}")
    optprint(f"Looking for configurations in: {configuration_directory_path}")
    optprint(f"Dimensional configuration option is {is_dimensional_configuration}")

    if args.single is not None:
        list_of_configs = [configuration_directory_path / args.single]
    else:
        list_of_configs = list(configuration_directory_path.glob("*.yaml")) + list(
            configuration_directory_path.glob("*.yml")
        )
    cfgs = []
    for config_path in list_of_configs:
        if is_dimensional_configuration:
            cfgs.append(get_config(DimensionalParams.load(config_path)))
        else:
            cfgs.append(Config.load(config_path))

    run_batch(cfgs, output_directory_path, verbosity_level=args.verbose)
