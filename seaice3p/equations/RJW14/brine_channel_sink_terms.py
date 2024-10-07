from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .brine_drainage import calculate_brine_channel_sink
from ...params import (
    Config,
    NoBrineConvection,
    MonoBubbleParams,
    PowerLawBubbleParams,
    EQMPhysicalParams,
    DISEQPhysicalParams,
)
from ..velocities.power_law_distribution import calculate_power_law_lag_factor
from ..velocities.mono_distribution import calculate_mono_lag_factor
from ...grids import geometric, Grids
from ...state import StateBCs, EQMStateBCs, DISEQStateBCs


def get_brine_convection_sink(
    cfg: Config, grids: Grids
) -> Callable[[StateBCs], NDArray]:
    fun_map = {
        EQMPhysicalParams: _EQM_brine_convection_sink,
        DISEQPhysicalParams: _DISEQ_brine_convection_sink,
    }

    def brine_convection_sink(state_BCs: StateBCs) -> NDArray:
        return fun_map[type(cfg.physical_params)](state_BCs, cfg, grids)

    return brine_convection_sink


def _EQM_brine_convection_sink(state_BCs: EQMStateBCs, cfg, grids) -> NDArray:
    """TODO: check the sink terms for bulk_dissolved_gas and gas fraction

    For now neglect the coupling of bubbles to the horizontal or vertical flow
    """
    heat_sink = _calculate_heat_sink(state_BCs, cfg, grids)
    salt_sink = _calculate_salt_sink(state_BCs, cfg, grids)
    gas_sink = _calculate_gas_sink(state_BCs, cfg, grids)
    return np.hstack((heat_sink, salt_sink, gas_sink))


def _DISEQ_brine_convection_sink(state_BCs: DISEQStateBCs, cfg, grids) -> NDArray:
    """TODO: check the sink terms for bulk_dissolved_gas and gas fraction

    For now neglect the coupling of bubbles to the horizontal or vertical flow
    """
    heat_sink = _calculate_heat_sink(state_BCs, cfg, grids)
    salt_sink = _calculate_salt_sink(state_BCs, cfg, grids)
    bulk_dissolved_gas_sink = _calculate_bulk_dissolved_gas_sink(state_BCs, cfg, grids)
    gas_fraction_sink = np.zeros_like(heat_sink)
    return np.hstack((heat_sink, salt_sink, bulk_dissolved_gas_sink, gas_fraction_sink))


def _calculate_heat_sink(state_BCs, cfg: Config, grids):
    liquid_fraction = state_BCs.liquid_fraction[1:-1]
    liquid_salinity = state_BCs.liquid_salinity[1:-1]
    temperature = state_BCs.temperature[1:-1]
    center_grid = grids.centers
    edge_grid = grids.edges

    if isinstance(cfg.brine_convection_params, NoBrineConvection):
        return np.zeros_like(liquid_fraction)

    sink = calculate_brine_channel_sink(
        liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg
    )
    return sink * temperature


def _calculate_salt_sink(state_BCs, cfg: Config, grids):
    liquid_fraction = state_BCs.liquid_fraction[1:-1]
    liquid_salinity = state_BCs.liquid_salinity[1:-1]
    center_grid = grids.centers
    edge_grid = grids.edges

    if isinstance(cfg.brine_convection_params, NoBrineConvection):
        return np.zeros_like(liquid_fraction)

    sink = calculate_brine_channel_sink(
        liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg
    )
    return sink * (liquid_salinity + cfg.physical_params.concentration_ratio)


def _calculate_gas_sink(state_BCs, cfg: Config, grids):
    """This is for the EQM model

    TODO: fix bug in bubble coupling to flow
    """
    liquid_fraction = state_BCs.liquid_fraction[1:-1]
    liquid_salinity = state_BCs.liquid_salinity[1:-1]
    dissolved_gas = state_BCs.dissolved_gas[1:-1]
    gas_fraction = state_BCs.gas_fraction[1:-1]
    center_grid = grids.centers
    edge_grid = grids.edges

    if isinstance(cfg.brine_convection_params, NoBrineConvection):
        return np.zeros_like(liquid_fraction)

    sink = calculate_brine_channel_sink(
        liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg
    )

    dissolved_gas_term = cfg.physical_params.expansion_coefficient * dissolved_gas

    if cfg.brine_convection_params.couple_bubble_to_horizontal_flow:
        if isinstance(cfg.bubble_params, MonoBubbleParams):
            lag_factor = calculate_mono_lag_factor(state_BCs.liquid_fraction, cfg)
        elif isinstance(cfg.bubble_params, PowerLawBubbleParams):
            lag_factor = calculate_power_law_lag_factor(state_BCs.liquid_fraction, cfg)
        else:
            raise NotImplementedError

        bubble_term = 2 * gas_fraction * geometric(lag_factor) / liquid_fraction
    else:
        bubble_term = np.zeros_like(liquid_fraction)

    return sink * (dissolved_gas_term + bubble_term)


def _calculate_bulk_dissolved_gas_sink(state_BCs, cfg: Config, grids):
    """This is for the DISEQ model"""
    liquid_fraction = state_BCs.liquid_fraction[1:-1]
    liquid_salinity = state_BCs.liquid_salinity[1:-1]
    dissolved_gas = state_BCs.dissolved_gas[1:-1]
    center_grid = grids.centers
    edge_grid = grids.edges

    if isinstance(cfg.brine_convection_params, NoBrineConvection):
        return np.zeros_like(liquid_fraction)

    sink = calculate_brine_channel_sink(
        liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg
    )

    return sink * cfg.physical_params.expansion_coefficient * dissolved_gas
