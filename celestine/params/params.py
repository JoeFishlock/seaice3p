"""Classes containing parameters required to run a simulation

The config class contains all the parameters needed to run a simulation as well
as methods to save and load this configuration to a yaml file."""

from pathlib import Path
from dataclasses import field
from serde import serde, coerce
from serde.yaml import from_yaml, to_yaml

from .forcing import BRW09Forcing, ForcingConfig
from .initial_conditions import InitialConditionsConfig, BRW09InitialConditions
from .numerical import NumericalParams
from .physical import PhysicalParams, EQMPhysicalParams
from .bubble import BubbleParams, MonoBubbleParams
from .convection import BrineConvectionParams, RJW14Params
from .convert import Scales


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
