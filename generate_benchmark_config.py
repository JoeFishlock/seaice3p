"""Generate yaml simulation config files for benchmarking different solvers
these are saved in benchmarks/

5 simulations all run for yearly forcing with small bubbles:
3 of implicit lax fredriech solver with three different resolutions
1 explicit lax friedrich solver
1 explicit lagged upwind solver
"""
import celestine.params

base_timestep_implicit = 5e-4
base_timestep_excplicit = 1.6e-4
base_I = 50

LowRes = celestine.params.NumericalParams(
    I=base_I, timestep=base_timestep_implicit, solver="LXFImplicit"
)
MedRes = celestine.params.NumericalParams(
    I=base_I * 2, timestep=base_timestep_implicit / 2, solver="LXFImplicit"
)
HighRes = celestine.params.NumericalParams(
    I=base_I * 4, timestep=base_timestep_implicit / 4, solver="LXFImplicit"
)
LXF = celestine.params.NumericalParams(
    I=base_I, timestep=base_timestep_excplicit, solver="LXF"
)
LU = celestine.params.NumericalParams(
    I=base_I, timestep=base_timestep_excplicit, solver="LU"
)

yearly_forcing_config = celestine.params.ForcingConfig(
    temperature_forcing_choice="yearly"
)

micro_bubbles = celestine.params.DarcyLawParams(bubble_radius_scaled=0.1)


solver_configurations = {
    "LU": LU,
    "LXF": LXF,
    "LXFLow": LowRes,
    "LXFMed": MedRes,
    "LXFHigh": HighRes,
}

for name, solver_configuration in solver_configurations.items():
    cfg = celestine.params.Config(
        name=name,
        total_time=4,
        savefreq=5e-2,
        data_path="benchmarks/",
        darcy_law_params=micro_bubbles,
        forcing_config=yearly_forcing_config,
        numerical_params=solver_configuration,
    )
    cfg.save()
