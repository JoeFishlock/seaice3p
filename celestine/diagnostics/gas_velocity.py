"""This script is used to plot the different gas interstitial velocities against
liquid fraction
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import celestine.velocities as vel
from celestine.grids import geometric, initialise_grids
from celestine.dimensional_params import DimensionalParams


def main(output_dir: Path):
    MAUS_THROAT = 3.89e-4 / 2
    MAUS_THROAT_POWER = 0.46

    output_dir.mkdir(exist_ok=True, parents=True)

    class MockStateBCs:
        def __init__(self, liquid_fraction):
            self.liquid_fraction = liquid_fraction
            self.liquid_salinity = np.zeros_like(liquid_fraction)
            I = np.size(liquid_fraction)
            _, centers, edges, _ = initialise_grids(I)
            self.grid = centers
            self.edge_grid = edges

    def calculate_volume_average_bubble_radius(max, min, p):
        cubed_radius = ((p - 1) * (max ** (4 - p) - min ** (4 - p))) / (
            (4 - p) * (min ** (1 - p) - max ** (1 - p))
        )
        return cubed_radius ** (1 / 3)

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
            porosity_threshold=True,
            porosity_threshold_value=0.024,
            I=1000,
        )
        cfg = dimensional_cfg.get_config()
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
    plt.savefig(output_dir / "Wall_drag_function.pdf")
    plt.close()

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
        plt.figure(figsize=(8, 8))
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
        for type, drag_exponent, LABEL, LINE in zip(
            3 * ["power"] + ["Haberman"], [1.5, 2.0, 2.5, None], LABELS, LINES
        ):
            cfg, dimensional_cfg = define_test_configurations(
                MAUS_THROAT,
                MAUS_THROAT_POWER,
                AVR_BUBBLE_SIZE,
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
                "k" + LINE,
                label=LABEL
                + f" Single average bubble volume size ={1000*AVR_BUBBLE_SIZE:.2g}mm",
            )

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

        plt.legend(prop={"size": 7})
        plt.xlabel("Liquid Fraction")
        plt.ylabel("Interstitial Gas Velocity (m/hour)")
        plt.yscale("log")
        plt.xscale("log")
        plt.title(
            f"p={BUBBLE_DISTRIBUTION_POWER}, max radius = {1000*MAX_BUBBLE_SIZE:.2g}mm, min radius = {1000*MIN_BUBBLE_SIZE:.2g}"
        )
        plt.savefig(output_dir / (name + ".pdf"))
        plt.close()

    generate_interstitial_gas_velocity_plot("Light Sizes", 1e-3, 1e-6, 1.5)
    generate_interstitial_gas_velocity_plot("Crabeck Sizes", 5e-4, 4.5e-5, 1.5)

    def calculate_terminal_bubble_velocity_moreau_2014(bubble_radius):
        """This accounts for transitiion to turbulent drag and they use a slightly
        different value for kinematic viscosity than my liquid density and dynamic
        viscosity"""
        chi = 9.81 * bubble_radius**3 / (2.7e-6**2)
        y = 10.82 / chi
        return ((2 * bubble_radius**2 * 9.81) / (9 * 2.7e-6)) * (
            np.sqrt(y**2 + 2 * y) - y
        )

    def calculate_free_slip_terminal_bubble_velocity(bubble_radius):
        return (bubble_radius**2 * 9.81) / (3 * 2.7e-6)

    def Haberman_function(L):
        output = (1 - 1.5 * L + 1.5 * L**5 - L**6) / (1 + 1.5 * L**5)
        output = np.where(L <= 0, 1, output)
        output = np.where(L >= 1, 0, output)
        return output

    def calculate_terminal_bubble_with_wall_drag(bubble_radius, liquid_fraction):
        tube_radius = MAUS_THROAT * liquid_fraction**MAUS_THROAT_POWER
        L = bubble_radius / tube_radius
        return calculate_free_slip_terminal_bubble_velocity(
            bubble_radius
        ) * Haberman_function(L)

    bubble_radius = np.linspace(1e-6, 5e-3, 1000)
    moreau = calculate_terminal_bubble_velocity_moreau_2014(bubble_radius)
    free_slip = calculate_free_slip_terminal_bubble_velocity(bubble_radius)
    plt.figure(figsize=(8, 8))
    plt.loglog(bubble_radius, moreau * 3600, "r", label="Moreau et al 2014")
    plt.loglog(bubble_radius, free_slip * 3600, "b", label="Free slip Stoke's")
    for phi in np.linspace(0.1, 1, 10):
        free_slip_drag = calculate_terminal_bubble_with_wall_drag(
            bubble_radius, liquid_fraction=phi
        )
        plt.loglog(
            bubble_radius, free_slip_drag * 3600, label=f"liquid fraction = {phi:.2f}"
        )
    plt.legend(prop={"size": 7})
    plt.xlabel("bubble radius (m)")
    plt.ylabel("terminal rise velocity (m/hour)")
    plt.savefig(output_dir / "terminal_velocity.pdf")
    plt.close()

    """Plot haberman drag power law bubble distribution interstitial velocities against
    liquid fraction for different parameter values"""
    BUBBLE_DISTRIBUTION_POWER = 1.5
    MINIMUM_BUBBLE_SIZE = 4.5e-5
    MAXIMUM_BUBBLE_SIZE = 5.5e-4
    plt.figure(figsize=(8, 8))
    maximum_bubble_sizes = np.geomspace(1e-4, 5e-3, 15)
    color_indices = np.linspace(0, 1, maximum_bubble_sizes.shape[0])
    # power law distributed bubble size
    for maximum_bubble_size, color_index in zip(maximum_bubble_sizes, color_indices):
        cfg, dimensional_cfg = define_test_configurations(
            MAUS_THROAT,
            MAUS_THROAT_POWER,
            maximum_bubble_size,
            2.0,
            "power_law",
            "Haberman",
            BUBBLE_DISTRIBUTION_POWER,
            maximum_bubble_size,
            MINIMUM_BUBBLE_SIZE,
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
            label=f"max bubble radius == {1000*maximum_bubble_size:.2e}mm",
            color=plt.cm.viridis(color_index),
        )

    plt.legend(prop={"size": 7})
    plt.xlabel("Liquid Fraction")
    plt.ylabel("Interstitial Gas Velocity (m/hour)")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Changing interstitial gas velocity curves with maximum bubble size")
    plt.savefig(output_dir / "Changing_maximum.pdf")
    plt.close()

    plt.figure(figsize=(8, 8))
    minimum_bubble_sizes = np.geomspace(1e-6, 1e-4, 15)
    color_indices = np.linspace(0, 1, minimum_bubble_sizes.shape[0])
    # power law distributed bubble size
    for minimum_bubble_size, color_index in zip(minimum_bubble_sizes, color_indices):
        cfg, dimensional_cfg = define_test_configurations(
            MAUS_THROAT,
            MAUS_THROAT_POWER,
            MAXIMUM_BUBBLE_SIZE,
            2.0,
            "power_law",
            "Haberman",
            BUBBLE_DISTRIBUTION_POWER,
            MAXIMUM_BUBBLE_SIZE,
            minimum_bubble_size,
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
            label=f"min bubble radius == {1000*minimum_bubble_size:.2e}mm",
            color=plt.cm.viridis(color_index),
        )

    plt.legend(prop={"size": 7})
    plt.xlabel("Liquid Fraction")
    plt.ylabel("Interstitial Gas Velocity (m/hour)")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Changing interstitial gas velocity curves with minimum bubble size")
    plt.savefig(output_dir / "Changing_minimum.pdf")
    plt.close()

    plt.figure(figsize=(8, 8))
    bubble_distribution_powers = np.linspace(1.1, 1.9, 9)
    color_indices = np.linspace(0, 1, bubble_distribution_powers.shape[0])
    # power law distributed bubble size
    for bubble_distribution_power, color_index in zip(
        bubble_distribution_powers, color_indices
    ):
        cfg, dimensional_cfg = define_test_configurations(
            MAUS_THROAT,
            MAUS_THROAT_POWER,
            MAXIMUM_BUBBLE_SIZE,
            2.0,
            "power_law",
            "Haberman",
            bubble_distribution_power,
            MAXIMUM_BUBBLE_SIZE,
            MINIMUM_BUBBLE_SIZE,
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
            label=f"bubble distribution power == {bubble_distribution_power:.1f}",
            color=plt.cm.viridis(color_index),
        )

    plt.legend(prop={"size": 7})
    plt.xlabel("Liquid Fraction")
    plt.ylabel("Interstitial Gas Velocity (m/hour)")
    plt.yscale("log")
    plt.xscale("log")
    plt.title(
        "Changing interstitial gas velocity curves with bubble distribution power"
    )
    plt.savefig(output_dir / "Changing_power.pdf")
    plt.close()


if __name__ == "__main__":
    OUTPUT_DIR = Path("gas_velocity_diagnostics")
    main(OUTPUT_DIR)
