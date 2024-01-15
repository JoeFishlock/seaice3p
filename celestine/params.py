"""Classes containing parameters required to run a simulation

The config class contains all the parameters needed to run a simulation as well
as methods to save and load this configuration to a yaml file."""

from yaml import safe_load, dump
from dataclasses import dataclass, asdict
import numpy as np
from celestine.logging_config import logger
from typing import ClassVar


@dataclass
class PhysicalParams:
    """non dimensional numbers for the mushy layer"""

    expansion_coefficient: float = 0.029
    concentration_ratio: float = 0.17
    stefan_number: float = 4.2
    lewis_salt: float = np.inf
    lewis_gas: float = np.inf
    frame_velocity: float = 0


def filter_missing_values(air_temp, days):
    """Filter out missing values are recorded as 9999"""
    is_missing = np.abs(air_temp) > 100
    return air_temp[~is_missing], days[~is_missing]


@dataclass
class BoundaryConditionsConfig:
    """values for bottom (ocean) boundary"""

    initial_conditions_choice: str = "uniform"
    far_gas_sat: float = 1.0
    far_temp: float = 0.1
    far_bulk_salinity: float = 0


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
class ForcingConfig:
    """choice of top boundary (atmospheric) forcing and required parameters"""

    temperature_forcing_choice: str = "constant"
    constant_top_temperature: float = -1.5
    offset: float = -1.0
    amplitude: float = 0.75
    period: float = 4.0

    Barrow_top_temperature_data_choice: str = "air"
    Barrow_initial_bulk_gas_in_ice: float = 1 / 5

    # class variables with barrow forcing data hard coded in
    DATA_INDEXES: ClassVar[dict[str, int]] = {
        "time": 0,
        "air": 8,
        "bottom_snow": 18,
        "top_ice": 19,
        "ocean": 43,
    }
    BARROW_DATA_PATH: ClassVar[str] = "celestine/forcing_data/BRW09.txt"

    def load_forcing_data(self):
        """populate class attributes with barrow dimensional air temperature
        and time in days (with missing values filtered out).

        Note the metadata explaining how to use the barrow temperature data is also
        in celestine/forcing_data. The indices corresponding to days and air temp are
        hard coded in as class variables.
        """
        data = np.genfromtxt(self.BARROW_DATA_PATH, delimiter="\t")
        top_temp_index = self.DATA_INDEXES[self.Barrow_top_temperature_data_choice]
        ocean_temp_index = self.DATA_INDEXES["ocean"]
        time_index = self.DATA_INDEXES["time"]

        barrow_top_temp = data[:, top_temp_index]
        barrow_days = data[:, time_index] - data[0, time_index]
        barrow_top_temp, barrow_days = filter_missing_values(
            barrow_top_temp, barrow_days
        )

        barrow_bottom_temp = data[:, ocean_temp_index]
        barrow_ocean_days = data[:, time_index] - data[0, time_index]
        barrow_bottom_temp, barrow_ocean_days = filter_missing_values(
            barrow_bottom_temp, barrow_ocean_days
        )

        self.barrow_top_temp = barrow_top_temp
        self.barrow_bottom_temp = barrow_bottom_temp
        self.barrow_ocean_days = barrow_ocean_days
        self.barrow_days = barrow_days


@dataclass
class NumericalParams:
    """parameters needed for discretisation and choice of numerical method"""

    I: int = 50
    timestep: float = 2e-4
    regularisation: float = 1e-6
    solver: str = "SCI"

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
    physical_params: PhysicalParams = PhysicalParams()
    boundary_conditions_config: BoundaryConditionsConfig = BoundaryConditionsConfig()
    darcy_law_params: DarcyLawParams = DarcyLawParams()
    forcing_config: ForcingConfig = ForcingConfig()
    numerical_params: NumericalParams = NumericalParams()
    scales: int = None
    total_time: float = 4.0
    savefreq: float = 5e-4  # save data after this amount of non-dimensional time
    data_path: str = "data/"

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

    def check_thermal_Courant_number(self):
        """Check if courant number for thermal diffusion term is low enough for
        explicit method and if it isn't log a warning.
        """
        if self.numerical_params.Courant > 0.5:
            logger.warning(f"Courant number is {self.numerical_params.Courant}")
