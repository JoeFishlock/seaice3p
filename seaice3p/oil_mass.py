from numpy.typing import NDArray


def convert_gas_fraction_to_oil_mass_ratio(
    gas_fraction: NDArray, oil_density: float, ice_density: float
) -> NDArray:
    """Convert gas (oil) volume fraction to oil mass ratio in ng/g"""
    return gas_fraction * 1e9 * oil_density / ice_density


def convert_oil_mass_ratio_to_gas_fraction(
    oil_mass_ratio: NDArray, oil_density: float, ice_density: float
) -> NDArray:
    """Convert oil mass ratio in ng/g to gas (oil) volume fraction"""
    return oil_mass_ratio * 1e-9 * ice_density / oil_density
