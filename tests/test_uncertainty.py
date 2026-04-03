"""Tests for uncertainty quantification across HYDE scenarios."""
import numpy as np
import pandas as pd
import pytest
from analysis.shared.uncertainty import (
    compute_scenario_stats,
    compute_manski_bounds,
)


def test_compute_scenario_stats(synthetic_scenario_panel):
    result = compute_scenario_stats(synthetic_scenario_panel)
    assert "mean" in result.columns
    assert "std" in result.columns
    assert "se" in result.columns
    assert "scenario" not in result.columns
    assert len(result) == len(synthetic_scenario_panel) // 3


def test_compute_scenario_stats_values():
    df = pd.DataFrame({
        "scenario": ["base", "lower", "upper"],
        "year": [1000, 1000, 1000],
        "region": ["Europe", "Europe", "Europe"],
        "var": ["pop", "pop", "pop"],
        "units": ["persons", "persons", "persons"],
        "value": [100.0, 80.0, 120.0],
    })
    result = compute_scenario_stats(df)
    row = result.iloc[0]
    np.testing.assert_allclose(row["mean"], 100.0)
    expected_std = np.std([100.0, 80.0, 120.0], ddof=0)
    np.testing.assert_allclose(row["std"], expected_std)
    np.testing.assert_allclose(row["se"], expected_std / np.sqrt(3))


def test_compute_manski_bounds():
    coefficients = {"base": -0.05, "lower": -0.08, "upper": -0.03}
    lb, ub = compute_manski_bounds(coefficients)
    assert lb == -0.08
    assert ub == -0.03
