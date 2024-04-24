import numpy as np
import celestine.params as cp
from ..enthalpy_method import DisequilibriumEnthalpyMethod
from .abstract_state import State
from .disequilibrium_state_bcs import DISEQStateBCs


class DISEQState(State):
    """Stores information needed for solution at one timestep on cell centers"""

    def __init__(
        self, cfg: cp.Config, time, enthalpy, salt, bulk_dissolved_gas, gas_fraction
    ):
        """Define bulk dissolved gas for the system as

        expansion_coefficient * liquid_fraction * dissolved_gas

        so that this is different from the dissolved gas concentration and

        bulk_gas = bulk_dissolved_gas + gas_fraction

        in non-dimensional units.
        """
        self.cfg = cfg
        self.time = time
        self.enthalpy = enthalpy
        self.salt = salt
        self.bulk_dissolved_gas = bulk_dissolved_gas
        self.gas_fraction = gas_fraction

    def get_state_with_bcs(self):
        """Initialise the appropriate StateBCs object"""
        return DISEQStateBCs(self)

    @classmethod
    def init_from_stacked_state(cls, cfg: cp.Config, time, stacked_state):
        """initialise from stacked solution vector for use in the solver"""
        enthalpy, salt, bulk_dissolved_gas, gas_fraction = np.split(stacked_state, 4)
        return cls(cfg, time, enthalpy, salt, bulk_dissolved_gas, gas_fraction)

    def get_stacked_state(self):
        return np.hstack(
            (self.enthalpy, self.salt, self.bulk_dissolved_gas, self.gas_fraction)
        )

    def calculate_enthalpy_method(self):
        (
            temperature,
            liquid_fraction,
            gas_fraction,
            solid_fraction,
            liquid_salinity,
            dissolved_gas,
        ) = DisequilibriumEnthalpyMethod(
            self.cfg.physical_params
        ).calculate_enthalpy_method(
            self
        )
        self.temperature = temperature
        self.liquid_fraction = liquid_fraction
        self.gas_fraction = gas_fraction
        self.solid_fraction = solid_fraction
        self.liquid_salinity = liquid_salinity
        self.dissolved_gas = dissolved_gas
