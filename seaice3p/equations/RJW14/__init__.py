"""Module to calculate the sink terms for conservation equations when using the
Rees Jones and Worster 2014 brine drainage parameterisation.

These terms represent loss through the brine channels and need to be added in the
convecting region when using this parameterisation
"""

from .brine_drainage import calculate_brine_convection_liquid_velocity
from .brine_channel_sink_terms import get_brine_convection_sink
