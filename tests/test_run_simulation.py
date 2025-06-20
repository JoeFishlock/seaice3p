import pytest
from glob import glob
from pathlib import Path
from seaice3p import (
    solve,
    DimensionalParams,
    Config,
    get_config,
    DimensionalBRW09Forcing,
    DimensionalBRW09OceanForcing,
    BRW09InitialConditions,
    DimensionalEQMGasParams,
    DimensionalDISEQGasParams,
    DimensionalRJW14Params,
    NoBrineConvection,
    DimensionalPowerLawBubbleParams,
    DimensionalMonoBubbleParams,
    NumericalParams,
)

COMMON_PARAMS = {
    "total_time_in_days": 1,
    "savefreq_in_days": 0.1,
    "lengthscale": 1,
}
BRINE = DimensionalRJW14Params(Rayleigh_critical=40, convection_strength=0.03)
NUM = NumericalParams(I=24)


@pytest.mark.parametrize(
    "simulation_parameters",
    [
        DimensionalParams(
            name="no_brine_convection",
            brine_convection_params=NoBrineConvection(),
            bubble_params=DimensionalMonoBubbleParams(),
            forcing_config=DimensionalBRW09Forcing(),
            ocean_forcing_config=DimensionalBRW09OceanForcing(),
            initial_conditions_config=BRW09InitialConditions(),
            gas_params=DimensionalEQMGasParams(),
            numerical_params=NUM,
            **COMMON_PARAMS
        ),
        DimensionalParams(
            name="power_law_bubbles",
            brine_convection_params=NoBrineConvection(),
            forcing_config=DimensionalBRW09Forcing(),
            ocean_forcing_config=DimensionalBRW09OceanForcing(),
            initial_conditions_config=BRW09InitialConditions(),
            gas_params=DimensionalEQMGasParams(),
            bubble_params=DimensionalPowerLawBubbleParams(),
            numerical_params=NUM,
            **COMMON_PARAMS
        ),
        DimensionalParams(
            name="brine_drainage_eqm",
            brine_convection_params=BRINE,
            forcing_config=DimensionalBRW09Forcing(),
            ocean_forcing_config=DimensionalBRW09OceanForcing(),
            initial_conditions_config=BRW09InitialConditions(),
            gas_params=DimensionalEQMGasParams(),
            bubble_params=DimensionalMonoBubbleParams(),
            numerical_params=NUM,
            **COMMON_PARAMS
        ),
        DimensionalParams(
            name="brine_drainage_diseq",
            brine_convection_params=BRINE,
            forcing_config=DimensionalBRW09Forcing(),
            ocean_forcing_config=DimensionalBRW09OceanForcing(),
            initial_conditions_config=BRW09InitialConditions(),
            gas_params=DimensionalDISEQGasParams(),
            bubble_params=DimensionalMonoBubbleParams(),
            numerical_params=NUM,
            **COMMON_PARAMS
        ),
    ],
)
def test_short_solve(tmp_path, simulation_parameters: DimensionalParams):
    solve(get_config(simulation_parameters), tmp_path)


@pytest.mark.slow
def test_best_barrow_config(tmp_path):
    solve(
        get_config(
            DimensionalParams.load(
                Path(
                    "tests/test_configurations/best_barrow/best_barrow_dimensional.yml"
                )
            )
        ),
        tmp_path,
    )


@pytest.mark.parametrize(
    "cfg_path",
    list(glob("tests/test_configurations/yearly_forcing/*.yml"))
    + list(glob("tests/test_configurations/constant_forcing/*.yml")),
)
@pytest.mark.slow
def test_yearly_and_constant_forcing_configurations(tmp_path, cfg_path: str):
    cfg = Config.load(Path(cfg_path))
    solve(cfg, tmp_path)
