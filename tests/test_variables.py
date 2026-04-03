"""Tests for derived Malthusian and intensification variables."""
import numpy as np
import pandas as pd
import pytest
from analysis.shared.variables import (
    compute_land_labor_ratio,
    compute_ag_output_proxy,
    compute_intensification_index,
    compute_pop_growth_rate,
    pivot_country_panel_wide,
)


def test_compute_land_labor_ratio():
    cropland = np.array([10.0, 20.0, 5.0])
    grazing = np.array([5.0, 10.0, 15.0])
    population = np.array([1e6, 2e6, 0.5e6])
    result = compute_land_labor_ratio(cropland, grazing, population)
    expected = (cropland + grazing) / population
    np.testing.assert_allclose(result, expected)


def test_compute_land_labor_ratio_zero_pop():
    result = compute_land_labor_ratio(
        np.array([10.0]), np.array([5.0]), np.array([0.0])
    )
    assert np.isnan(result[0])


def test_compute_ag_output_proxy():
    cropland = np.array([10.0])
    grazing = np.array([6.0])
    result = compute_ag_output_proxy(cropland, grazing, grazing_weight=0.5)
    np.testing.assert_allclose(result, [13.0])


def test_compute_intensification_index():
    irrigation_share = np.array([0.2, 0.0, 0.5])
    cropland = np.array([10.0, 0.0, 8.0])
    grazing = np.array([5.0, 10.0, 2.0])
    result = compute_intensification_index(irrigation_share, cropland, grazing)
    expected = np.array([0.2 * 10 / 15, 0.0, 0.5 * 8 / 10])
    np.testing.assert_allclose(result, expected)


def test_compute_intensification_index_zero_land():
    result = compute_intensification_index(
        np.array([0.5]), np.array([0.0]), np.array([0.0])
    )
    assert np.isnan(result[0])


def test_compute_pop_growth_rate():
    pop = pd.Series([100.0, 110.0, 121.0, 133.1])
    years = pd.Series([1000, 1100, 1200, 1300])
    result = compute_pop_growth_rate(pop, years)
    assert len(result) == 4
    assert np.isnan(result.iloc[0])
    np.testing.assert_allclose(result.iloc[1], (110 / 100) ** (1 / 100) - 1, rtol=1e-6)


def test_pivot_country_panel_wide(synthetic_country_panel):
    wide = pivot_country_panel_wide(synthetic_country_panel)
    assert "pop_persons_mean" in wide.columns
    assert "cropland_mha_mean" in wide.columns
    assert "year" in wide.columns
    assert "country" in wide.columns
