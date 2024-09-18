"""Module containing enthalpy method to calculate state variables from bulk enthalpy,
bulk salinity and bulk gas."""

from typing import Callable
from ..params import Config, EQMPhysicalParams, DISEQPhysicalParams
from .phase_boundaries import get_phase_masks
from ..state import EQMState, DISEQState, State, StateFull, EQMStateFull, DISEQStateFull
from .gas import (
    calculate_EQM_gas_fraction,
    calculate_DISEQ_dissolved_gas,
    calculate_EQM_dissolved_gas,
)
from .common import calculate_common_enthalpy_method_vars


def get_enthalpy_method(cfg: Config) -> Callable[[State], StateFull]:
    fun_map = {
        EQMPhysicalParams: _calculate_EQM_enthalpy_method,
        DISEQPhysicalParams: _calculate_DISEQ_enthalpy_method,
    }

    def enthalpy_method(state: State) -> StateFull:
        return fun_map[type(cfg.physical_params)](state, cfg)

    return enthalpy_method


def _calculate_EQM_enthalpy_method(state: EQMState, cfg: Config) -> StateFull:
    phase_masks = get_phase_masks(state, cfg.physical_params)
    (
        solid_fraction,
        liquid_fraction,
        temperature,
        liquid_salinity,
    ) = calculate_common_enthalpy_method_vars(state, cfg, phase_masks)
    dissolved_gas = calculate_EQM_dissolved_gas(
        state, liquid_fraction, cfg.physical_params
    )

    gas_fraction = calculate_EQM_gas_fraction(
        state, liquid_fraction, cfg.physical_params
    )
    return EQMStateFull(
        state.time,
        state.enthalpy,
        state.salt,
        state.gas,
        temperature,
        liquid_fraction,
        solid_fraction,
        liquid_salinity,
        dissolved_gas,
        gas_fraction,
    )


def _calculate_DISEQ_enthalpy_method(state: DISEQState, cfg: Config) -> StateFull:
    phase_masks = get_phase_masks(state, cfg.physical_params)
    (
        solid_fraction,
        liquid_fraction,
        temperature,
        liquid_salinity,
    ) = calculate_common_enthalpy_method_vars(state, cfg, phase_masks)
    dissolved_gas = calculate_DISEQ_dissolved_gas(
        state, liquid_fraction, cfg.physical_params, phase_masks
    )

    return DISEQStateFull(
        state.time,
        state.enthalpy,
        state.salt,
        state.bulk_dissolved_gas,
        state.gas_fraction,
        temperature,
        liquid_fraction,
        solid_fraction,
        liquid_salinity,
        dissolved_gas,
    )
