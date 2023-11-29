import numpy as np
from celestine.brine_drainage import calculate_brine_channel_sink
from celestine.params import Config
from celestine.velocities import (
    calculate_mono_lag_factor,
    calculate_power_law_lag_factor,
)
from celestine.grids import geometric


def calculate_heat_sink(state_BCs, cfg: Config):
    liquid_fraction = state_BCs.liquid_fraction[1:-1]
    liquid_salinity = state_BCs.liquid_salinity[1:-1]
    temperature = state_BCs.temperature[1:-1]
    center_grid = state_BCs.grid[1:-1]
    edge_grid = state_BCs.edge_grid

    if not cfg.darcy_law_params.brine_convection_parameterisation:
        return np.zeros_like(liquid_fraction)

    sink = calculate_brine_channel_sink(
        liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg
    )
    return sink * temperature


def calculate_salt_sink(state_BCs, cfg: Config):
    liquid_fraction = state_BCs.liquid_fraction[1:-1]
    liquid_salinity = state_BCs.liquid_salinity[1:-1]
    center_grid = state_BCs.grid[1:-1]
    edge_grid = state_BCs.edge_grid

    if not cfg.darcy_law_params.brine_convection_parameterisation:
        return np.zeros_like(liquid_fraction)

    sink = calculate_brine_channel_sink(
        liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg
    )
    return sink * (liquid_salinity + cfg.physical_params.concentration_ratio)


def calculate_gas_sink(state_BCs, cfg: Config):
    liquid_fraction = state_BCs.liquid_fraction[1:-1]
    liquid_salinity = state_BCs.liquid_salinity[1:-1]
    dissolved_gas = state_BCs.dissolved_gas[1:-1]
    gas_fraction = state_BCs.gas_fraction[1:-1]
    center_grid = state_BCs.grid[1:-1]
    edge_grid = state_BCs.edge_grid

    if not cfg.darcy_law_params.brine_convection_parameterisation:
        return np.zeros_like(liquid_fraction)

    sink = calculate_brine_channel_sink(
        liquid_fraction, liquid_salinity, center_grid, edge_grid, cfg
    )

    dissolved_gas_term = cfg.physical_params.expansion_coefficient * dissolved_gas

    if cfg.darcy_law_params.couple_bubble_to_horizontal_flow:
        if cfg.darcy_law_params.bubble_size_distribution_type == "mono":
            lag_factor = calculate_mono_lag_factor(liquid_fraction, cfg)
        elif cfg.darcy_law_params.bubble_size_distribution_type == "power_law":
            lag_factor = calculate_power_law_lag_factor(liquid_fraction, cfg)
        else:
            raise ValueError(
                f"Bubble size distribution of type {cfg.darcy_law_params.bubble_size_distribution_type} not recognised"
            )
        bubble_term = 2 * gas_fraction * geometric(lag_factor) / liquid_fraction
    else:
        bubble_term = np.zeros_like(liquid_fraction)

    return sink * (dissolved_gas_term + bubble_term)
