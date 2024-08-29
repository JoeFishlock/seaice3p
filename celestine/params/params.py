"""Classes containing parameters required to run a simulation

The config class contains all the parameters needed to run a simulation as well
as methods to save and load this configuration to a yaml file."""

from pathlib import Path
from dataclasses import field
from serde import serde, coerce
from serde.yaml import from_yaml, to_yaml

from .forcing import BRW09Forcing, ForcingConfig, get_dimensionless_forcing_config
from .initial_conditions import (
    InitialConditionsConfig,
    BRW09InitialConditions,
    get_dimensionless_initial_conditions_config,
)
from .physical import (
    PhysicalParams,
    EQMPhysicalParams,
    get_dimensionless_physical_params,
)
from .bubble import BubbleParams, MonoBubbleParams, get_dimensionless_bubble_params
from .convection import (
    BrineConvectionParams,
    RJW14Params,
    get_dimensionless_brine_convection_params,
)
from .convert import Scales
from .dimensional import DimensionalParams, NumericalParams


@serde(type_check=coerce)
class Config:
    """contains all information needed to run a simulation and save output

    this config object can be saved and loaded to a yaml file."""

    name: str
    total_time: float = 4.0
    savefreq: float = 5e-4  # save data after this amount of non-dimensional time

    physical_params: PhysicalParams = field(default_factory=EQMPhysicalParams)
    bubble_params: BubbleParams = field(default_factory=MonoBubbleParams)
    brine_convection_params: BrineConvectionParams = field(default_factory=RJW14Params)
    forcing_config: ForcingConfig = field(default_factory=BRW09Forcing)
    initial_conditions_config: InitialConditionsConfig = field(
        default_factory=BRW09InitialConditions
    )
    numerical_params: NumericalParams = field(default_factory=NumericalParams)
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
    return Config(
        name=dimensional_params.name,
        physical_params=physical_params,
        initial_conditions_config=initial_conditions_config,
        brine_convection_params=brine_convection_params,
        bubble_params=bubble_params,
        forcing_config=forcing_config,
        numerical_params=dimensional_params.numerical_params,
        scales=dimensional_params.scales,
        total_time=dimensional_params.total_time,
        savefreq=dimensional_params.savefreq,
    )
