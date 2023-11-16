"""This script is used to plot the different gas interstitial velocities against
liquid fraction
"""

import numpy as np
import matplotlib.pyplot as plt
import celestine.velocities as vel
from celestine.grids import get_difference_matrix, geometric
from celestine.params import Config, DarcyLawParams, NumericalParams
from celestine.dimensional_params import DimensionalParams


if __name__ == "__main__":
    MAUS_THROAT = 3.95e-4
    MAUS_THROAT_POWER = 0.466

    class MockStateBCs:
        def __init__(self, liquid_fraction):
            self.liquid_fraction = liquid_fraction
            self.pressure = np.zeros_like(liquid_fraction)

    def calculate_volume_average_bubble_radius(max, min, p):
        cubed_radius = ((p - 1) * (max ** (4 - p) - min ** (4 - p))) / (
            (4 - p) * (min ** (1 - p) - max ** (1 - p))
        )
        return cubed_radius ** (1 / 3)

    # mono = Config(
    #     "mono-disperse",
    #     darcy_law_params=DarcyLawParams(
    #         B=2e7,
    #         bubble_radius_scaled=0.5,
    #         pore_throat_scaling=0.5,
    #         drag_exponent=2.5,
    #         liquid_velocity=0.0,
    #     ),
    # )

    # power_law = Config(
    #     "poly-disperse",
    #     darcy_law_params=DarcyLawParams(
    #         B=2e7,
    #         bubble_size_distribution_type="power_law",
    #         pore_throat_scaling=0.5,
    #         drag_exponent=2.5,
    #         liquid_velocity=0.0,
    #         bubble_distribution_power=1.5,
    #         maximum_bubble_radius_scaled=1,
    #         minimum_bubble_radius_scaled=1e-3,
    #     ),
    # )
    # mono_Haberman = Config(
    #     "mono-disperse-Haberman",
    #     darcy_law_params=DarcyLawParams(
    #         B=2e7,
    #         bubble_radius_scaled=0.5,
    #         pore_throat_scaling=0.5,
    #         drag_exponent=2.5,
    #         liquid_velocity=0.0,
    #         wall_drag_law_choice="Haberman",
    #     ),
    # )

    # power_law_Haberman = Config(
    #     "poly-disperse-Haberman",
    #     darcy_law_params=DarcyLawParams(
    #         B=2e7,
    #         bubble_size_distribution_type="power_law",
    #         pore_throat_scaling=0.5,
    #         drag_exponent=2.5,
    #         liquid_velocity=0.0,
    #         bubble_distribution_power=1.5,
    #         maximum_bubble_radius_scaled=1,
    #         minimum_bubble_radius_scaled=1e-3,
    #         wall_drag_law_choice="Haberman",
    #     ),
    # )
    # liquid_fraction = np.linspace(0, 1, mono.numerical_params.I + 2)
    # D_g = get_difference_matrix(mono.numerical_params.I + 1, mono.numerical_params.step)
    # mono_vel, _, _ = vel.calculate_velocities(
    #     state_BCs=MockStateBCs(liquid_fraction=liquid_fraction),
    #     cfg=mono,
    # )
    # poly_vel, _, _ = vel.calculate_velocities(
    #     state_BCs=MockStateBCs(liquid_fraction=liquid_fraction),
    #     cfg=power_law,
    # )
    # mono_vel_Haberman, _, _ = vel.calculate_velocities(
    #     state_BCs=MockStateBCs(liquid_fraction=liquid_fraction),
    #     cfg=mono_Haberman,
    # )
    # poly_vel_Haberman, _, _ = vel.calculate_velocities(
    #     state_BCs=MockStateBCs(liquid_fraction=liquid_fraction),
    #     cfg=power_law_Haberman,
    # )
    # plt.figure()
    # plt.plot(geometric(liquid_fraction), mono_vel, "r*-", label=mono.name)
    # plt.plot(geometric(liquid_fraction), poly_vel, "b*-", label=power_law.name)
    # plt.plot(
    #     geometric(liquid_fraction), mono_vel_Haberman, "r*--", label=mono_Haberman.name
    # )
    # plt.plot(
    #     geometric(liquid_fraction),
    #     poly_vel_Haberman,
    #     "b*--",
    #     label=power_law_Haberman.name,
    # )
    # plt.legend()
    # plt.show()

    def define_test_configurations(
        pore_radius,
        pore_throat_scaling,
        bubble_radius,
        drag_exponent,
        bubble_size_distribution_type,
        wall_drag_law_choice,
        bubble_distribution_power,
        maxmimum_bubble_radius,
        minimum_bubble_radius,
    ):

        dimensional_cfg = DimensionalParams(
            name="config",
            total_time_in_days=164,
            savefreq_in_days=3,
            lengthscale=1.0,
            pore_radius=pore_radius,
            pore_throat_scaling=pore_throat_scaling,
            bubble_radius=bubble_radius,
            drag_exponent=drag_exponent,
            bubble_size_distribution_type=bubble_size_distribution_type,
            wall_drag_law_choice=wall_drag_law_choice,
            bubble_distribution_power=bubble_distribution_power,
            maximum_bubble_radius=maxmimum_bubble_radius,
            minimum_bubble_radius=minimum_bubble_radius,
        )
        cfg = dimensional_cfg.get_config(
            numerical_params=NumericalParams(solver="SCI", I=100),
        )
        return cfg, dimensional_cfg

    """Plot wall drag enhancement against bubble size fraction for the case where we
    fit a recirpocal power law compared with the fit in Haberman and Sayre 1958"""
    plt.figure()
    LABELS = [
        "power law r=1.5",
        "power law r=2",
        "power law r=2.5",
        "Haberman and Sayre 1958",
    ]
    for type, drag_exponent, LABEL in zip(
        3 * ["power"] + ["Haberman"], [1.5, 2.0, 2.5, None], LABELS
    ):
        cfg, dimensional_cfg = define_test_configurations(
            MAUS_THROAT,
            MAUS_THROAT_POWER,
            0.1e-3,
            drag_exponent,
            "mono",
            type,
            1.5,
            1e-3,
            1e-6,
        )
        bubble_size_fraction = np.linspace(0, 1, cfg.numerical_params.I)
        wall_drag = vel.calculate_wall_drag_function(bubble_size_fraction, cfg)
        plt.plot(bubble_size_fraction, wall_drag, label=LABEL)
    plt.legend()
    plt.xlabel("Bubble size fraction")
    plt.ylabel("1 / wall drag enhancement function")
    plt.title("Wall drag as function of bubble radius to tube radius")
    plt.savefig("Wall_drag_function.pdf")

    """Plot gas interstitial velocities against liquid fraciton for the different power
    law drag exponents and Haberman as different linestyles.

    Do three cases each with a different colour:
    Mono bubble size with maximum bubble size
    Mono bubble size with volume average bubble size
    Power law distributed bubble size
    """

    def generate_interstitial_gas_velocity_plot(
        name, MAX_BUBBLE_SIZE, MIN_BUBBLE_SIZE, BUBBLE_DISTRIBUTION_POWER
    ):
        plt.figure()
        LABELS = [
            "power law r=1.5",
            "power law r=2",
            "power law r=2.5",
            "Haberman and Sayre 1958",
        ]
        LINES = [":", "-.", "--", "-"]
        AVR_BUBBLE_SIZE = calculate_volume_average_bubble_radius(
            MAX_BUBBLE_SIZE, MIN_BUBBLE_SIZE, BUBBLE_DISTRIBUTION_POWER
        )

        # mono distributed bubble with maximum bubble size
        for type, drag_exponent, LABEL, LINE in zip(
            3 * ["power"] + ["Haberman"], [1.5, 2.0, 2.5, None], LABELS, LINES
        ):
            cfg, dimensional_cfg = define_test_configurations(
                MAUS_THROAT,
                MAUS_THROAT_POWER,
                MAX_BUBBLE_SIZE,
                drag_exponent,
                "mono",
                type,
                BUBBLE_DISTRIBUTION_POWER,
                MAX_BUBBLE_SIZE,
                MIN_BUBBLE_SIZE,
            )
            liquid_fraction = np.linspace(0, 1, cfg.numerical_params.I + 2)
            mock_state = MockStateBCs(liquid_fraction)
            Vg, _, _ = vel.calculate_velocities(mock_state, cfg)

            # convert Vg to be in m/day
            scales = dimensional_cfg.get_scales()
            conversion_factor = scales.velocity_scale_in_m_per_day / 24
            Vg = Vg * conversion_factor

            plt.plot(
                geometric(liquid_fraction),
                Vg,
                "r" + LINE,
                label=LABEL
                + f" Single maximum bubble size ={1000*MAX_BUBBLE_SIZE:.2g}mm",
            )

        # mono distributed bubble with average volume bubble size
        # for type, drag_exponent, LABEL, LINE in zip(
        #     3 * ["power"] + ["Haberman"], [1.5, 2.0, 2.5, None], LABELS, LINES
        # ):
        #     cfg, dimensional_cfg = define_test_configurations(
        #         MAUS_THROAT,
        #         MAUS_THROAT_POWER,
        #         AVR_BUBBLE_SIZE,
        #         drag_exponent,
        #         "mono",
        #         type,
        #         BUBBLE_DISTRIBUTION_POWER,
        #         MAX_BUBBLE_SIZE,
        #         MIN_BUBBLE_SIZE,
        #     )
        #     liquid_fraction = np.linspace(0, 1, cfg.numerical_params.I + 2)
        #     mock_state = MockStateBCs(liquid_fraction)
        #     Vg, _, _ = vel.calculate_velocities(mock_state, cfg)

        #     # convert Vg to be in m/day
        #     scales = dimensional_cfg.get_scales()
        #     conversion_factor = scales.velocity_scale_in_m_per_day / 24
        #     Vg = Vg * conversion_factor

        #     plt.plot(
        #         geometric(liquid_fraction),
        #         Vg,
        #         "k" + LINE,
        #         label=LABEL
        #         + f" Single average bubble volume size ={1000*AVR_BUBBLE_SIZE:.2g}mm",
        #     )

        # power law distributed bubble size
        for type, drag_exponent, LABEL, LINE in zip(
            3 * ["power"] + ["Haberman"], [1.5, 2.0, 2.5, None], LABELS, LINES
        ):
            cfg, dimensional_cfg = define_test_configurations(
                MAUS_THROAT,
                MAUS_THROAT_POWER,
                MAX_BUBBLE_SIZE,
                drag_exponent,
                "power_law",
                type,
                BUBBLE_DISTRIBUTION_POWER,
                MAX_BUBBLE_SIZE,
                MIN_BUBBLE_SIZE,
            )
            liquid_fraction = np.linspace(0, 1, cfg.numerical_params.I + 2)
            mock_state = MockStateBCs(liquid_fraction)
            Vg, _, _ = vel.calculate_velocities(mock_state, cfg)

            # convert Vg to be in m/day
            scales = dimensional_cfg.get_scales()
            conversion_factor = scales.velocity_scale_in_m_per_day / 24
            Vg = Vg * conversion_factor

            plt.plot(
                geometric(liquid_fraction),
                Vg,
                "b" + LINE,
                label=LABEL + f" power law bubble size distribution",
            )

        plt.legend()
        plt.xlabel("Liquid Fraction")
        plt.ylabel("Interstitial Gas Velocity (m/hour)")
        plt.title(
            f"p={BUBBLE_DISTRIBUTION_POWER}, max radius = {1000*MAX_BUBBLE_SIZE:.2g}mm, min radius = {1000*MIN_BUBBLE_SIZE:.2g}"
        )
        plt.savefig(name + ".pdf")

    generate_interstitial_gas_velocity_plot("throat_power1.5", 1e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_power1.2", 1e-3, 1e-5, 1.2)
    generate_interstitial_gas_velocity_plot("throat_power1.7", 1e-3, 1e-5, 1.7)

    generate_interstitial_gas_velocity_plot("throat_max01", 0.1e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max02", 0.2e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max03", 0.3e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max04", 0.4e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max05", 0.5e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max06", 0.6e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max07", 0.7e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max08", 0.8e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max09", 0.9e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max1", 1e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max11", 1.1e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max12", 1.2e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max13", 1.3e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max14", 1.4e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_max15", 1.5e-3, 1e-5, 1.5)

    generate_interstitial_gas_velocity_plot("throat_min5", 1e-3, 1e-5, 1.5)
    generate_interstitial_gas_velocity_plot("throat_min55", 1e-3, 5e-6, 1.5)
    generate_interstitial_gas_velocity_plot("throat_min6", 1e-3, 1e-6, 1.5)
