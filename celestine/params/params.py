"""Classes containing parameters required to run a simulation

The config class contains all the parameters needed to run a simulation as well
as methods to save and load this configuration to a yaml file."""

from pathlib import Path
from dataclasses import dataclass, asdict, field
from serde import serde, coerce
from serde.yaml import from_yaml, to_yaml
import numpy as np

from .forcing import BRW09Forcing, ForcingConfig
from .initial_conditions import InitialConditionsConfig, BRW09InitialConditions
from .numerical import NumericalParams
from .convert import Scales


@serde(type_check=coerce)
class PhysicalParams:
    """non dimensional numbers for the mushy layer"""

    expansion_coefficient: float = 0.029
    concentration_ratio: float = 0.17
    stefan_number: float = 4.2
    lewis_salt: float = np.inf
    lewis_gas: float = np.inf
    frame_velocity: float = 0

    # Option to average the conductivity term.
    phase_average_conductivity: bool = False
    conductivity_ratio: float = 4.11

    # Option to change tolerable supersaturation
    tolerable_super_saturation_fraction: float = 1

    # only used in DISEQ model
    damkohler_number: float = 1


@serde(type_check=coerce)
class DarcyLawParams:
    """non dimensional parameters for calculating liquid and gas darcy velocities"""

    B: float = 100
    pore_throat_scaling: float = 1 / 2
    bubble_size_distribution_type: str = "mono"
    wall_drag_law_choice: str = "power"

    # needed for power fit wall drag function
    drag_exponent: float = 6.0

    # for mono size distribution
    bubble_radius_scaled: float = 1.0

    # for power law size distribution
    bubble_distribution_power: float = 1.5
    minimum_bubble_radius_scaled: float = 1e-3
    maximum_bubble_radius_scaled: float = 1

    porosity_threshold: bool = False
    porosity_threshold_value: float = 0.024

    brine_convection_parameterisation: bool = False
    Rayleigh_salt: float = 44105
    Rayleigh_critical: float = 40
    convection_strength: float = 0.03

    couple_bubble_to_horizontal_flow: bool = True
    couple_bubble_to_vertical_flow: bool = True


@serde(type_check=coerce)
class Config:
    """contains all information needed to run a simulation and save output

    this config object can be saved and loaded to a yaml file."""

    name: str
    model: str = "EQM"
    physical_params: PhysicalParams = field(default_factory=PhysicalParams)
    initial_conditions_config: InitialConditionsConfig = field(
        default_factory=BRW09InitialConditions
    )
    darcy_law_params: DarcyLawParams = field(default_factory=DarcyLawParams)
    forcing_config: ForcingConfig = field(default_factory=BRW09Forcing)
    numerical_params: NumericalParams = field(default_factory=NumericalParams)
    scales: Scales | None = None
    total_time: float = 4.0
    savefreq: float = 5e-4  # save data after this amount of non-dimensional time

    def save(self, directory: Path):
        with open(directory / f"{self.name}.yml", "w") as outfile:
            outfile.write(to_yaml(self))

    @classmethod
    def load(cls, path):
        with open(path, "r") as infile:
            yaml = infile.read()
        return from_yaml(cls, yaml)
