"""Tests for paper4_shadow seasonality computation module."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper4_shadow.seasonality import (
    compute_intra_annual_seasonality_era5,
    compute_historical_seasonality_proxy,
    build_seasonality_panel,
)


def _make_monthly_df(amplitude, region="A", years=range(2000, 2010)):
    """Helper: create monthly temperature data with given amplitude."""
    rows = []
    for y in years:
        for m in range(1, 13):
            temp = 15 + amplitude * np.sin(2 * np.pi * m / 12)
            rows.append({"region": region, "year": y, "month": m, "temperature_c": temp})
    return pd.DataFrame(rows)


def test_sinusoidal_seasonality_ordering():
    """Region with amplitude=11 should have higher seasonality than amplitude=2."""
    df_high = _make_monthly_df(amplitude=11, region="high")
    df_low = _make_monthly_df(amplitude=2, region="low")
    df = pd.concat([df_high, df_low], ignore_index=True)
    result = compute_intra_annual_seasonality_era5(df)
    mean_high = result.loc[result["region"] == "high", "seasonality_temp"].mean()
    mean_low = result.loc[result["region"] == "low", "seasonality_temp"].mean()
    assert mean_high > mean_low


def test_era5_output_columns():
    """Output should have expected columns."""
    df = _make_monthly_df(amplitude=5)
    result = compute_intra_annual_seasonality_era5(df)
    assert "region" in result.columns
    assert "year" in result.columns
    assert "seasonality_temp" in result.columns


def test_era5_with_precipitation():
    """When precipitation_mm is present, seasonality_precip should appear."""
    df = _make_monthly_df(amplitude=5)
    df["precipitation_mm"] = 50 + 30 * np.sin(2 * np.pi * df["month"] / 12)
    result = compute_intra_annual_seasonality_era5(df)
    assert "seasonality_precip" in result.columns


def test_historical_proxy_volatile_vs_stable():
    """More volatile region should have higher seasonality proxy."""
    rng = np.random.default_rng(42)
    years = np.arange(1, 200)
    stable = pd.DataFrame({
        "region": "stable", "year": years,
        "temperature_c": 15 + rng.normal(0, 0.3, len(years)),
    })
    volatile = pd.DataFrame({
        "region": "volatile", "year": years,
        "temperature_c": 15 + rng.normal(0, 3.0, len(years)),
    })
    df = pd.concat([stable, volatile], ignore_index=True)
    result = compute_historical_seasonality_proxy(df, window=50)
    valid = result.dropna(subset=["seasonality_proxy"])
    mean_vol = valid.loc[valid["region"] == "volatile", "seasonality_proxy"].mean()
    mean_stb = valid.loc[valid["region"] == "stable", "seasonality_proxy"].mean()
    assert mean_vol > mean_stb


def test_build_panel_one_row_per_region():
    """build_seasonality_panel should return one row per region."""
    rng = np.random.default_rng(99)
    years = np.arange(1, 300)
    regions = ["A", "B", "C"]
    rows = []
    for r in regions:
        for y in years:
            rows.append({"region": r, "year": y, "temperature_c": 15 + rng.normal()})
    annual = pd.DataFrame(rows)
    result = build_seasonality_panel(annual, historical_period=(1, 200))
    assert len(result) == len(regions)
    assert set(result["region"]) == set(regions)
