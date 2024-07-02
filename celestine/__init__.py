__version__ = "0.14.0"

# Exported functions and classes
from .params import (
    Config,
    NumericalParams,
    DarcyLawParams,
    ForcingConfig,
    BoundaryConditionsConfig,
    PhysicalParams,
)
from .dimensional_params import DimensionalParams
from .run_simulation import solve, run_batch
from .state import get_model
from .forcing import get_bottom_temperature_forcing, get_temperature_forcing
from .grids import Grids
from .RJW14.brine_drainage import calculate_ice_ocean_boundary_depth
from .load import load_data, get_state, get_array_data
