import pytest
from celestine import solve
from celestine import DimensionalParams
from celestine.params.dimensional import (
    DimensionalBRW09Forcing,
    BRW09InitialConditions,
    DimensionalEQMGasParams,
    DimensionalDISEQGasParams,
    DimensionalRJW14Params,
    NoBrineConvection,
)
from celestine.params.initial_conditions import UniformInitialConditions
from celestine.params.numerical import NumericalParams

COMMON_PARAMS = {
    "total_time_in_days": 1,
    "savefreq_in_days": 0.1,
}
BRINE = DimensionalRJW14Params(Rayleigh_critical=40, convection_strength=0.03)
NUM = NumericalParams(I=24)


@pytest.mark.parametrize(
    "simulation_parameters",
    [
        DimensionalParams(
            name="no_brine_convection",
            brine_convection_params=NoBrineConvection(),
            forcing_config=DimensionalBRW09Forcing(),
            initial_conditions_config=BRW09InitialConditions(),
            gas_params=DimensionalEQMGasParams(),
            numerical_params=NUM,
            **COMMON_PARAMS
        ),
        DimensionalParams(
            name="brine_drainage_eqm",
            brine_convection_params=BRINE,
            forcing_config=DimensionalBRW09Forcing(),
            initial_conditions_config=BRW09InitialConditions(),
            gas_params=DimensionalEQMGasParams(),
            numerical_params=NUM,
            **COMMON_PARAMS
        ),
        DimensionalParams(
            name="brine_drainage_diseq",
            brine_convection_params=BRINE,
            forcing_config=DimensionalBRW09Forcing(),
            initial_conditions_config=BRW09InitialConditions(),
            gas_params=DimensionalDISEQGasParams(),
            numerical_params=NUM,
            **COMMON_PARAMS
        ),
    ],
)
def test_short_solve(tmp_path, simulation_parameters: DimensionalParams):
    solve(simulation_parameters.get_config(), tmp_path)
