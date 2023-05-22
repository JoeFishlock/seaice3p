"""Physical (dimensional) parameters required to run a simulation and convert output
to dimensional variables.

The DimensionalParams class contains all the dimensional parameters needed to produce
a simulation configuration.

The Scales class contains all the dimensional parameters required to convert simulation
output between physical and non-dimensional variables.
"""

from yaml import safe_load, dump
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class DimensionalParams:
    name: str
    total_time: float = 365  # days
    savefreq: float = 1  # save data after this amount of time in days
    data_path: str = "data/"

    def save(self):
        with open(f"{self.data_path}{self.name}.yml", "w") as outfile:
            dump(asdict(self), outfile)

    @classmethod
    def load(cls, path):
        with open(path, "r") as infile:
            dictionary = safe_load(infile)
        return cls(**dictionary)
