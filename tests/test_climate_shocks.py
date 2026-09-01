"""Tests for ERA5 climate shock construction."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper3_climate.climate_shocks import compute_anomalies, compute_volatility

def test_compute_anomalies():
    years = np.arange(1950, 2020)
    values = np.sin(np.arange(70) * 0.3) * 2 + 15
    df = pd.DataFrame({"year": years, "temperature": values})
    result = compute_anomalies(df, "temperature", rolling_window=30)
    valid = result["temperature_anomaly"].dropna()
    assert abs(valid.mean()) < 1.0
    assert "temperature_anomaly" in result.columns

def test_compute_volatility():
    years = np.arange(1950, 2020)
    rng = np.random.default_rng(42)
    values = rng.normal(15, 2, size=70)
    df = pd.DataFrame({"year": years, "temperature": values})
    result = compute_volatility(df, "temperature", rolling_window=10)
    assert "temperature_volatility" in result.columns
    valid = result["temperature_volatility"].dropna()
    assert 0.5 < valid.mean() < 5.0

def test_build_climate_shock_panel():
    from analysis.paper3_climate.climate_shocks import build_climate_shock_panel

    # Create sample panel data for two regions over several years
    years = np.arange(1950, 2020)

    region1 = pd.DataFrame({
        "region": "A",
        "year": years,
        "temp": np.sin(np.arange(70) * 0.3) * 2 + 15,
        "precip": np.cos(np.arange(70) * 0.3) * 5 + 100
    })

    region2 = pd.DataFrame({
        "region": "B",
        "year": years,
        "temp": np.sin(np.arange(70) * 0.4) * 2 + 20,
        "precip": np.cos(np.arange(70) * 0.4) * 5 + 50
    })

    panel = pd.concat([region1, region2], ignore_index=True)

    # Run the function
    result = build_climate_shock_panel(panel, ["temp", "precip"], entity_col="region", rolling_window=30)

    # Verify the structure
    assert "temp_anomaly" in result.columns
    assert "temp_volatility" in result.columns
    assert "precip_anomaly" in result.columns
    assert "precip_volatility" in result.columns

    # Length should match input
    assert len(result) == len(panel)

    # Regions should still be A and B
    assert set(result["region"]) == {"A", "B"}

    # We should have valid values for the non-NaN part of the rolling window
    valid_A = result[result["region"] == "A"]["temp_anomaly"].dropna()
    valid_B = result[result["region"] == "B"]["temp_anomaly"].dropna()

    # Anomaly mean should be close to 0
    assert abs(valid_A.mean()) < 1.0
    assert abs(valid_B.mean()) < 1.0


def test_build_climate_shock_panel_empty():
    from analysis.paper3_climate.climate_shocks import build_climate_shock_panel
    df = pd.DataFrame(columns=["region", "year", "temp", "precip"])

    with pytest.raises(ValueError):
        # pd.concat on an empty list throws a ValueError
        build_climate_shock_panel(df, ["temp", "precip"], entity_col="region")
