import numpy as np
from ..forcing import boundary_conditions as bc
from .equilibrium_state import EQMState, EQMStateFull, EQMStateBCs
from .disequilibrium_state import DISEQState, DISEQStateFull, DISEQStateBCs

State = EQMState | DISEQState
StateFull = EQMStateFull | DISEQStateFull
StateBCs = EQMStateBCs | DISEQStateBCs


def get_model(cfg) -> State:
    MODEL_CHOICES = {"EQM": EQMState, "DISEQ": DISEQState}
    return MODEL_CHOICES[cfg.model]


def get_state(cfg, time, solution_vector) -> State:
    model_choice = cfg.model
    match model_choice:
        case "EQM":
            enthalpy, salt, gas = np.split(solution_vector, 3)
            return EQMState(time, enthalpy, salt, gas)
        case "DISEQ":
            enthalpy, salt, bulk_dissolved_gas, gas_fraction = np.split(
                solution_vector, 4
            )
            return DISEQState(time, enthalpy, salt, bulk_dissolved_gas, gas_fraction)
        case _:
            raise NotImplementedError


def apply_boundary_conditions(cfg, full_state: StateFull) -> StateBCs:
    time = full_state.time
    enthalpy = bc.enthalpy_BCs(full_state.enthalpy, cfg)
    salt = bc.salt_BCs(full_state.salt, cfg)

    temperature = bc.temperature_BCs(full_state, time, cfg)
    liquid_salinity = bc.liquid_salinity_BCs(full_state.liquid_salinity, cfg)
    dissolved_gas = bc.dissolved_gas_BCs(full_state.dissolved_gas, cfg)
    gas_fraction = bc.gas_fraction_BCs(full_state.gas_fraction, cfg)
    liquid_fraction = bc.liquid_fraction_BCs(full_state.liquid_fraction, cfg)

    match full_state:
        case EQMStateFull():
            gas = bc.gas_BCs(full_state.gas, cfg)
            return EQMStateBCs(
                time,
                enthalpy,
                salt,
                gas,
                temperature,
                liquid_salinity,
                dissolved_gas,
                gas_fraction,
                liquid_fraction,
            )
        case DISEQStateFull():
            bulk_dissolved_gas = (
                cfg.physical_params.expansion_coefficient
                * liquid_fraction
                * dissolved_gas
            )
            return DISEQStateBCs(
                time,
                enthalpy,
                salt,
                temperature,
                liquid_salinity,
                dissolved_gas,
                liquid_fraction,
                bulk_dissolved_gas,
                gas_fraction,
            )
        case _:
            raise NotImplementedError
