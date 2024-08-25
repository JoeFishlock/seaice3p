from pathlib import Path
from serde import serde, coerce
import numpy as np


def _filter_missing_values(air_temp, days):
    """Filter out missing values are recorded as 9999"""
    is_missing = np.abs(air_temp) > 100
    return air_temp[~is_missing], days[~is_missing]


@serde(type_check=coerce)
class ConstantForcing:
    """Constant temperature forcing"""

    constant_top_temperature: float = -1.5

    ocean_gas_sat: float = 1.0
    ocean_temp: float = 0.1
    ocean_bulk_salinity: float = 0


@serde(type_check=coerce)
class YearlyForcing:
    """Yearly sinusoidal temperature forcing"""

    offset: float = -1.0
    amplitude: float = 0.75
    period: float = 4.0

    ocean_gas_sat: float = 1.0
    ocean_temp: float = 0.1
    ocean_bulk_salinity: float = 0


@serde(type_check=coerce)
class BRW09Forcing:
    """Surface and ocean temperature data loaded from thermistor temperature record
    during the Barrow 2009 field study.
    """

    Barrow_top_temperature_data_choice: str = "air"
    Barrow_initial_bulk_gas_in_ice: float = 1 / 5

    ocean_gas_sat: float = 1.0
    ocean_temp: float = 0.1
    ocean_bulk_salinity: float = 0

    def __post_init__(self):
        """populate class attributes with barrow dimensional air temperature
        and time in days (with missing values filtered out).

        Note the metadata explaining how to use the barrow temperature data is also
        in celestine/forcing_data. The indices corresponding to days and air temp are
        hard coded in as class variables.
        """
        DATA_INDICES = {
            "time": 0,
            "air": 8,
            "bottom_snow": 18,
            "top_ice": 19,
            "ocean": 43,
        }
        data = np.genfromtxt(
            Path(__file__).parent.parent / "forcing_data/BRW09.txt", delimiter="\t"
        )
        top_temp_index = DATA_INDICES[self.Barrow_top_temperature_data_choice]
        ocean_temp_index = DATA_INDICES["ocean"]
        time_index = DATA_INDICES["time"]

        barrow_top_temp = data[:, top_temp_index]
        barrow_days = data[:, time_index] - data[0, time_index]
        barrow_top_temp, barrow_days = _filter_missing_values(
            barrow_top_temp, barrow_days
        )

        barrow_bottom_temp = data[:, ocean_temp_index]
        barrow_ocean_days = data[:, time_index] - data[0, time_index]
        barrow_bottom_temp, barrow_ocean_days = _filter_missing_values(
            barrow_bottom_temp, barrow_ocean_days
        )

        self.barrow_top_temp = barrow_top_temp
        self.barrow_bottom_temp = barrow_bottom_temp
        self.barrow_ocean_days = barrow_ocean_days
        self.barrow_days = barrow_days


@serde(type_check=coerce)
class RadForcing:
    """Forcing parameters for radiative transfer simulation with oil drops"""

    surface_energy_balance_forcing: bool = True

    SW_internal_heating: bool = False
    SW_forcing_choice: str = "constant"
    constant_SW_irradiance: float = 280  # W/m2

    SW_radiation_model_choice: str = "1L"  # specify oilrad model to use
    # Parameters for single layer SW radiative transfer model
    constant_oil_mass_ratio: float = 0  # ng/g
    SW_scattering_ice_type: str = "FYI"

    ocean_gas_sat: float = 1.0
    ocean_temp: float = 0.1
    ocean_bulk_salinity: float = 0


ForcingConfig = ConstantForcing | YearlyForcing | BRW09Forcing | RadForcing
