import json
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Params:
    name: str
    expansion_coefficient: float = 0.029
    B: float = 100
    concentration_ratio: float = 0.17
    stefan_number: float = 4.2
    regularisation: float = 1e-6
    lewis_salt: float = np.inf
    lewis_gas: float = np.inf
    far_gas_sat: float = 1.0
    far_temp: float = 0.1
    I: int = 50
    total_time: float = 4.0
    timestep: float = 2e-4
    savefreq: float = 5e-4  # save data after this amount of non-dimensional time
    solver: str = "euler"
    data_path: str = "data/"
    frame_velocity: float = 0
    pore_throat_scaling: float = 1 / 2
    bubble_radius_scaled: float = 1.0
    drag_exponent: float = 6.0
    liquid_velocity: float = 0.0
    temperature_forcing_choice: str = "constant"
    constant_top_temperature: float = -1.5

    @property
    def step(self):
        return 1 / self.I

    @property
    def Diff_num(self):
        """This number must be <0.5 for stability of temperature diffusion terms"""
        return self.timestep / (self.step**2)

    @property
    def Courant_gas(self):
        """ "This number must be <1.0 for CFL condition of gas buoyant transport"""
        return self.B * self.timestep / self.step

    def save(self):
        with open(f"{self.data_path}{self.name}.json", "w") as fp:
            json.dump(asdict(self), fp)

    @classmethod
    def load(cls, path_to_json):
        with open(f"{path_to_json}", "r") as fp:
            dictionary = json.load(fp)
        return cls(**dictionary)

    def check_buoyancy_CFL(self):
        if self.Courant_gas > 1.0:
            return True
        else:
            return False

    def check_thermal_diffusion_stability(self):
        if self.Diff_num > 0.5:
            return True
        else:
            return False
