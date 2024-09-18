"""Test running the diagnostic scripts in celestine

These scripts plot functions used in the code like the gas velocities and brine
convection parameteristaion

These are useful for making plots and also as integration tests for these parts of
the code however they do produce file output
"""
import pytest


def test_running_brine_convection_diagnostics(tmp_path):
    from celestine.diagnostics.brine_drainage_parameterisation import main

    main(tmp_path)
