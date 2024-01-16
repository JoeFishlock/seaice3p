import argparse
from pathlib import Path
from celestine import __version__
from celestine.dimensional_params import DimensionalParams
from celestine.params import Config
from celestine.run_simulation import run_batch


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
        default="simulation_output",
    )
    parser.add_argument(
        "-d",
        "--dimensional",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="This flag makes the program expect configurations with dimensional parameters",
    )

    args = parser.parse_args()
    is_dimensional_configuration = args.dimensional
    configuration_directory_path = Path(args.configuration_directory)
    output_directory_path = Path(args.output_directory)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    list_of_configs = list(configuration_directory_path.glob("*.yaml")) + list(
        configuration_directory_path.glob("*.yml")
    )

    print(f"Running celestine version: {__version__}")
    print(f"Save simulation output to: {output_directory_path}")
    print(f"Looking for configurations in: {configuration_directory_path}")
    print(f"Dimensional configuration option is {is_dimensional_configuration}")

    cfgs = []
    for config_path in list_of_configs:
        if is_dimensional_configuration:
            cfgs.append(DimensionalParams.load(config_path).get_config())
        else:
            cfgs.append(Config.load(config_path))

    run_batch(cfgs, output_directory_path)
