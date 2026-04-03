"""Tests for data loading functions."""
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from unittest.mock import patch
from analysis.shared.loaders import (
    read_esri_ascii_grid,
    load_nc_variable,
    load_existing_country_panel,
    load_existing_scenario_panel,
    normalize_longitudes,
)


def test_normalize_longitudes_360_to_180():
    lon = np.array([0.5, 179.5, 180.5, 359.5])
    result = normalize_longitudes(lon)
    expected = np.array([-179.5, -0.5, 0.5, 179.5])
    # After sorting, should be -180 to 180 range
    assert result.min() >= -180.0
    assert result.max() <= 180.0


def test_normalize_longitudes_already_180():
    lon = np.array([-179.5, -0.5, 0.5, 179.5])
    result = normalize_longitudes(lon)
    np.testing.assert_array_equal(result, lon)


def test_load_existing_country_panel(tmp_path):
    df = pd.DataFrame({
        "year": [1000, 1000],
        "country": ["FRA", "FRA"],
        "var": ["pop_persons", "cropland_mha"],
        "units": ["persons", "Mha"],
        "mean": [1e6, 0.5],
        "std": [1e5, 0.05],
    })
    path = tmp_path / "country.csv"
    df.to_csv(path, index=False)
    result = load_existing_country_panel(path)
    assert list(result.columns) == ["year", "country", "var", "units", "mean", "std"]
    assert len(result) == 2


def test_load_existing_scenario_panel(tmp_path):
    df = pd.DataFrame({
        "scenario": ["base", "lower"],
        "year": [1000, 1000],
        "region": ["Europe", "Europe"],
        "var": ["pop", "pop"],
        "units": ["persons", "persons"],
        "value": [1e6, 0.85e6],
    })
    path = tmp_path / "scenario.csv"
    df.to_csv(path, index=False)
    result = load_existing_scenario_panel(path)
    assert "scenario" in result.columns
    assert len(result) == 2
