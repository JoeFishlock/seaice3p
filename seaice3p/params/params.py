"""Classes containing parameters required to run a simulation

The config class contains all the parameters needed to run a simulation as well
as methods to save and load this configuration to a yaml file."""

from pathlib import Path
from dataclasses import dataclass
from serde import serde, coerce
from serde.yaml import from_yaml, to_yaml

from .ocean_forcing import OceanForcingConfig, get_dimensionless_ocean_forcing_config
from .forcing import ForcingConfig, get_dimensionless_forcing_config
from .initial_conditions import (
    InitialConditionsConfig,
    get_dimensionless_initial_conditions_config,
)
from .physical import (
    PhysicalParams,
    get_dimensionless_physical_params,
)
from .bubble import BubbleParams, get_dimensionless_bubble_params
from .convection import (
    BrineConvectionParams,
    get_dimensionless_brine_convection_params,
)
from .convert import Scales
from .dimensional import DimensionalParams, NumericalParams


@serde(type_check=coerce)
@dataclass(frozen=True)
class Config:
    """contains all information needed to run a simulation and save output

    this config object can be saved and loaded to a yaml file."""

    name: str
    total_time: float
    savefreq: float

    physical_params: PhysicalParams
    bubble_params: BubbleParams
    brine_convection_params: BrineConvectionParams
    forcing_config: ForcingConfig
    ocean_forcing_config: OceanForcingConfig
    initial_conditions_config: InitialConditionsConfig
    numerical_params: NumericalParams = NumericalParams()
    scales: Scales | None = None

    def save(self, directory: Path):
        with open(directory / f"{self.name}.yml", "w") as outfile:
            outfile.write(to_yaml(self))

    @classmethod
    def load(cls, path):
        with open(path, "r") as infile:
            yaml = infile.read()
        return from_yaml(cls, yaml)


def get_config(dimensional_params: DimensionalParams) -> Config:
    """Return a Config object for the simulation.

    physical parameters and Darcy law parameters are calculated from the dimensional
    input. You can modify the numerical parameters and boundary conditions and
    forcing provided for the simulation."""
    physical_params = get_dimensionless_physical_params(dimensional_params)
    initial_conditions_config = get_dimensionless_initial_conditions_config(
        dimensional_params
    )
    brine_convection_params = get_dimensionless_brine_convection_params(
        dimensional_params
    )
    bubble_params = get_dimensionless_bubble_params(dimensional_params)
    forcing_config = get_dimensionless_forcing_config(dimensional_params)
    ocean_forcing_config = get_dimensionless_ocean_forcing_config(dimensional_params)
    return Config(
        name=dimensional_params.name,
        physical_params=physical_params,
        initial_conditions_config=initial_conditions_config,
        brine_convection_params=brine_convection_params,
        bubble_params=bubble_params,
        forcing_config=forcing_config,
        ocean_forcing_config=ocean_forcing_config,
        numerical_params=dimensional_params.numerical_params,
        scales=dimensional_params.scales,
        total_time=dimensional_params.total_time,
        savefreq=dimensional_params.savefreq,
    )
