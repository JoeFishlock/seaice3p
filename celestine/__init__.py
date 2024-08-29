__version__ = "0.14.0"

# Exported functions and classes
from .params import (
    Config,
    get_config,
    BubbleParams,
    BrineConvectionParams,
    NumericalParams,
    ForcingConfig,
    InitialConditionsConfig,
    PhysicalParams,
    DimensionalParams,
)
from .run_simulation import solve, run_batch
from .grids import Grids, calculate_ice_ocean_boundary_depth
from .load import load_data, get_state, get_array_data
