"""Generate yaml simulation config files for manual test cases.
"""
import celestine.params

TEST_DATA_DIR = "test_data/"

if __name__ == "__main__":
    yearly_forcing_config = celestine.params.ForcingConfig(
        temperature_forcing_choice="yearly"
    )
    constant_forcing_config = celestine.params.ForcingConfig(
        temperature_forcing_choice="constant"
    )

    macro_bubbles = celestine.params.DarcyLawParams(bubble_radius_scaled=1.0)
    medium_bubbles = celestine.params.DarcyLawParams(bubble_radius_scaled=0.5)
    micro_bubbles = celestine.params.DarcyLawParams(bubble_radius_scaled=0.1)

    LU_solver = celestine.params.NumericalParams(solver="LU")

    forcing_configurations = {
        "Yearly": yearly_forcing_config,
        "Constant": constant_forcing_config,
    }
    bubble_sizes = {
        "Macro": macro_bubbles,
        "Med": medium_bubbles,
        "Micro": micro_bubbles,
    }

    for forcing_string, forcing_config in forcing_configurations.items():
        for bubble_string, bubble_size in bubble_sizes.items():
            cfg = celestine.params.Config(
                name=forcing_string + bubble_string + "LU",
                total_time=4,
                savefreq=5e-2,
                data_path=TEST_DATA_DIR,
                darcy_law_params=bubble_size,
                forcing_config=forcing_config,
                numerical_params=LU_solver,
            )
            cfg.save()
