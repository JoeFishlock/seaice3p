"""Module to calculate Darcy velocities.

The liquid Darcy velocity must be parameterised.

The gas Darcy velocity is calculated as gas_fraction x interstitial bubble velocity

Interstitial bubble velocity is found by a steady state Stoke's flow calculation.
We have implemented two cases
mono: All bubbles nucleate and remain the same size
power_law: A power law bubble size distribution with fixed max and min.
"""

from .velocities import calculate_velocities
