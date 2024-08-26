"""Classes containing parameters required to run a simulation

The config class contains all the parameters needed to run a simulation as well
as methods to save and load this configuration to a yaml file."""

from pathlib import Path
from dataclasses import dataclass, asdict, field
from serde import serde, coerce
from serde.yaml import from_yaml, to_yaml

from .forcing import BRW09Forcing, ForcingConfig
from .initial_conditions import InitialConditionsConfig, BRW09InitialConditions
from .numerical import NumericalParams
from .physical import PhysicalParams, EQMPhysicalParams
from .convert import Scales


@serde(type_check=coerce)
class DarcyLawParams:
    """non dimensional parameters for calculating liquid and gas darcy velocities"""

    B: float = 100
    pore_throat_scaling: float = 1 / 2
    porosity_threshold: bool = False
    porosity_threshold_value: float = 0.024

    bubble_size_distribution_type: str = "mono"
    # for mono size distribution
    bubble_radius_scaled: float = 1.0
    # for power law size distribution
    bubble_distribution_power: float = 1.5
    minimum_bubble_radius_scaled: float = 1e-3
    maximum_bubble_radius_scaled: float = 1

    brine_convection_parameterisation: bool = False
    Rayleigh_salt: float = 44105
    Rayleigh_critical: float = 40
    convection_strength: float = 0.03
    couple_bubble_to_horizontal_flow: bool = False
    couple_bubble_to_vertical_flow: bool = False


@serde(type_check=coerce)
class Config:
    """contains all information needed to run a simulation and save output

    this config object can be saved and loaded to a yaml file."""

    name: str
    model: str = "EQM"
    total_time: float = 4.0
    savefreq: float = 5e-4  # save data after this amount of non-dimensional time

    physical_params: PhysicalParams = field(default_factory=EQMPhysicalParams)
    initial_conditions_config: InitialConditionsConfig = field(
        default_factory=BRW09InitialConditions
    )
    darcy_law_params: DarcyLawParams = field(default_factory=DarcyLawParams)
    forcing_config: ForcingConfig = field(default_factory=BRW09Forcing)
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
