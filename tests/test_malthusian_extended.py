"""Tests for extended Malthusian regressions."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper4_shadow.malthusian_extended import run_malthusian_by_pathway, run_rolling_by_pathway

@pytest.fixture
def malthusian_panel():
    rng = np.random.default_rng(42)
    rows = []
    # Need enough data points to successfully run regressions.
    # run_fe_regression needs valid data, so the variance can't be exactly 0, etc.
    for country in ["FRA", "GBR", "ESP", "POR", "CHN", "IND", "JPN", "KOR"]:
        for year in range(1000, 1900, 50): # many time points
            rows.append({
                "year": year,
                "country": country,
                "pop_growth": rng.normal(0.01, 0.05),
                "log_density": rng.normal(2, 1),
                "temp_volatility": rng.normal(0.5, 0.1),
                "log_land_labor": rng.normal(5, 1)
            })
    return pd.DataFrame(rows)

@pytest.fixture
def pathway_assignments():
    return pd.DataFrame({
        "country": ["FRA", "GBR", "ESP", "POR", "CHN", "IND", "JPN", "KOR"],
        "cluster": ["Early", "Early", "Early", "Early", "Late", "Late", "Late", "Late"]
    })

def test_run_malthusian_by_pathway(malthusian_panel, pathway_assignments):
    result = run_malthusian_by_pathway(
        malthusian_panel, pathway_assignments
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Two clusters
    assert "pathway" in result.columns
    assert "beta_density" in result.columns
    assert "pval_density" in result.columns
    assert "nobs" in result.columns
    assert "rsquared" in result.columns
    assert "gamma_volatility" in result.columns
    assert "pval_volatility" in result.columns

    assert set(result["pathway"]) == {"Early", "Late"}

def test_run_malthusian_by_pathway_missing_volatility(malthusian_panel, pathway_assignments):
    # Remove volatility column
    malthusian_panel_no_vol = malthusian_panel.drop(columns=["temp_volatility"])
    result = run_malthusian_by_pathway(
        malthusian_panel_no_vol, pathway_assignments
    )

    assert isinstance(result, pd.DataFrame)
    assert "gamma_volatility" not in result.columns
    assert "pval_volatility" not in result.columns

def test_run_rolling_by_pathway(malthusian_panel, pathway_assignments):
    result = run_rolling_by_pathway(
        malthusian_panel, pathway_assignments,
        window_years=400, step_years=100
    )

    assert isinstance(result, pd.DataFrame)
    assert "center_year" in result.columns
    assert "coefficient" in result.columns
    assert "pvalue" in result.columns
    assert "nobs" in result.columns
    assert "pathway" in result.columns

    assert set(result["pathway"]) == {"Early", "Late"}


def test_run_malthusian_by_pathway_regression_failure(malthusian_panel, pathway_assignments):
    # Set all values to nan so regression fails
    malthusian_panel.loc[:, "pop_growth"] = np.nan
    result = run_malthusian_by_pathway(malthusian_panel, pathway_assignments)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

def test_run_rolling_by_pathway_empty_frames(malthusian_panel, pathway_assignments):
    # Cause empty frames by using window_years greater than data span
    result = run_rolling_by_pathway(
        malthusian_panel, pathway_assignments,
        window_years=5000, step_years=100
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert "center_year" in result.columns
    assert "pathway" in result.columns
