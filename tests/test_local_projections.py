"""Tests for local projection impulse response functions."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper3_climate.local_projections import run_local_projection


@pytest.fixture
def climate_land_panel():
    rng = np.random.default_rng(42)
    rows = []
    for region in ["Europe", "East Asia", "South Asia"]:
        for year in range(1950, 2020):
            shock = rng.normal(0, 1)
            rows.append({"year": year, "region": region, "temp_anomaly": shock, "cropland_share": 0.3 + 0.02 * shock + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_run_local_projection(climate_land_panel):
    irf = run_local_projection(climate_land_panel, shock_var="temp_anomaly", response_var="cropland_share", entity_col="region", max_horizon=5)
    assert len(irf) == 6
    assert "horizon" in irf.columns
    assert "coefficient" in irf.columns
    assert "ci_lower" in irf.columns
    assert "ci_upper" in irf.columns


def test_local_projection_horizon_zero(climate_land_panel):
    irf = run_local_projection(climate_land_panel, shock_var="temp_anomaly", response_var="cropland_share", entity_col="region", max_horizon=0)
    assert abs(irf.iloc[0]["coefficient"] - 0.02) < 0.01
