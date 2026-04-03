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
