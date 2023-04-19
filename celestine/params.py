"""Classes containing parameters required to run a simulation

The config class contains all the parameters needed to run a simulation as well
as methods to save and load this configuration to a yaml file."""
from yaml import safe_load, dump
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class PhysicalParams:
    """non dimensional numbers for the mushy layer"""

    expansion_coefficient: float = 0.029
    concentration_ratio: float = 0.17
    stefan_number: float = 4.2
    lewis_salt: float = np.inf
    lewis_gas: float = np.inf
    frame_velocity: float = 0


@dataclass
class BoundaryConditionsConfig:
    """values for bottom (ocean) boundary"""

    far_gas_sat: float = 1.0
    far_temp: float = 0.1


@dataclass
class DarcyLawParams:
    """non dimensional parameters for calculating liquid and gas darcy velocities"""

    B: float = 100
    bubble_radius_scaled: float = 1.0
    pore_throat_scaling: float = 1 / 2
    drag_exponent: float = 6.0
    liquid_velocity: float = 0.0


@dataclass
class ForcingConfig:
    """choice of top boundary (atmospheric) forcing and required parameters"""

    temperature_forcing_choice: str = "constant"
    constant_top_temperature: float = -1.5
    offset: float = -1.0
    amplitude: float = 0.75
    period: float = 4.0


@dataclass
class NumericalParams:
    """parameters needed for discretisation and choice of numerical method"""

    I: int = 50
    timestep: float = 2e-4
    regularisation: float = 1e-6
    adaptive_timestepping: bool = False
    CFL_limit = 0.5
    Courant_limit = 0.4
    solver: str = "LXF"

    @property
    def step(self):
        return 1 / self.I

    @property
    def Diff_num(self):
        """This number must be <0.5 for stability of temperature diffusion terms"""
        return self.timestep / (self.step**2)


@dataclass
class Config:
    """contains all information needed to run a simulation and save output

    this config object can be saved and loaded to a yaml file."""

    name: str
    physical_params: PhysicalParams = PhysicalParams()
    boundary_conditions_config: BoundaryConditionsConfig = BoundaryConditionsConfig()
    darcy_law_params: DarcyLawParams = DarcyLawParams()
    forcing_config: ForcingConfig = ForcingConfig()
    numerical_params: NumericalParams = NumericalParams()
    total_time: float = 4.0
    savefreq: float = 5e-4  # save data after this amount of non-dimensional time
    data_path: str = "data/"

    @property
    def Courant_gas(self):
        """ "This number must be <1.0 for CFL condition of gas buoyant transport"""
        return (
            self.darcy_law_params.B
            * self.numerical_params.step
            / self.numerical_params.step
        )

    def save(self):
        with open(f"{self.data_path}{self.name}.yml", "w") as outfile:
            dump(asdict(self), outfile)

    @classmethod
    def load(cls, path):
        with open(path, "r") as infile:
            dictionary = safe_load(infile)
        return cls(
            name=dictionary["name"],
            total_time=dictionary["total_time"],
            savefreq=dictionary["savefreq"],
            data_path=dictionary["data_path"],
            physical_params=PhysicalParams(**dictionary["physical_params"]),
            boundary_conditions_config=BoundaryConditionsConfig(
                **dictionary["boundary_conditions_config"]
            ),
            darcy_law_params=DarcyLawParams(**dictionary["darcy_law_params"]),
            forcing_config=ForcingConfig(**dictionary["forcing_config"]),
            numerical_params=NumericalParams(**dictionary["numerical_params"]),
        )

    def check_buoyancy_CFL(self):
        if self.Courant_gas > 1.0:
            return True
        else:
            return False

    def check_thermal_diffusion_stability(self):
        if self.numerical_params.Diff_num > 0.5:
            return True
        else:
            return False
