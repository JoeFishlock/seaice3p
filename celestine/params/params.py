"""Classes containing parameters required to run a simulation

The config class contains all the parameters needed to run a simulation as well
as methods to save and load this configuration to a yaml file."""

from pathlib import Path
from dataclasses import dataclass, asdict, field
from yaml import safe_load, dump
import numpy as np

from .forcing import ForcingConfig


@dataclass
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


@dataclass
class BoundaryConditionsConfig:
    """values for bottom (ocean) boundary"""

    initial_conditions_choice: str = "uniform"
    far_gas_sat: float = 1.0
    far_temp: float = 0.1
    far_bulk_salinity: float = 0

    # Non dimensional parameters for summer initial conditions
    initial_summer_ice_depth: float = 0.5
    initial_summer_ocean_temperature: float = -0.05
    initial_summer_ice_temperature: float = -0.1


@dataclass
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


@dataclass
class NumericalParams:
    """parameters needed for discretisation and choice of numerical method"""

    I: int = 50
    timestep: float = 2e-4
    regularisation: float = 1e-6

    @property
    def step(self):
        return 1 / self.I

    @property
    def Courant(self):
        """This number must be <0.5 for stability of temperature diffusion terms"""
        return self.timestep / (self.step**2)


@dataclass
class Config:
    """contains all information needed to run a simulation and save output

    this config object can be saved and loaded to a yaml file."""

    name: str
    model: str = "EQM"
    physical_params: PhysicalParams = field(default_factory=PhysicalParams)
    boundary_conditions_config: BoundaryConditionsConfig = field(
        default_factory=BoundaryConditionsConfig
    )
    darcy_law_params: DarcyLawParams = field(default_factory=DarcyLawParams)
    forcing_config: ForcingConfig = field(default_factory=ForcingConfig)
    numerical_params: NumericalParams = field(default_factory=NumericalParams)
    scales: int = None
    total_time: float = 4.0
    savefreq: float = 5e-4  # save data after this amount of non-dimensional time

    def save(self, directory: Path):
        with open(directory / f"{self.name}.yml", "w") as outfile:
            dump(asdict(self), outfile)

    @classmethod
    def load(cls, path):
        with open(path, "r") as infile:
            dictionary = safe_load(infile)
        return cls(
            name=dictionary["name"],
            model=dictionary["model"],
            total_time=dictionary["total_time"],
            savefreq=dictionary["savefreq"],
            physical_params=PhysicalParams(**dictionary["physical_params"]),
            boundary_conditions_config=BoundaryConditionsConfig(
                **dictionary["boundary_conditions_config"]
            ),
            darcy_law_params=DarcyLawParams(**dictionary["darcy_law_params"]),
            forcing_config=ForcingConfig(**dictionary["forcing_config"]),
            numerical_params=NumericalParams(**dictionary["numerical_params"]),
        )

    def check_thermal_Courant_number(self):
        """Check if courant number for thermal diffusion term is low enough for
        explicit method and if it isn't log a warning.
        """
        if self.numerical_params.Courant > 0.5:
            print(f"Courant number is {self.numerical_params.Courant}")
