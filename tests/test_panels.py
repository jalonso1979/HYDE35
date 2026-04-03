"""Tests for Malthusian panel construction."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper2_malthus.panels import build_malthusian_panel

@pytest.fixture
def wide_panel():
    rows = []
    for country in ["FRA", "GBR"]:
        for year in range(1000, 1600, 100):
            rows.append({
                "year": year, "country": country,
                "pop_persons_mean": 1e6 * (1 + year / 1000),
                "popdens_p_km2_mean": 10 * (1 + year / 2000),
                "land_labor_ratio": 0.01 * (2 - year / 2000),
                "ag_output_proxy_mha": 5 * (1 + year / 1500),
                "intensification_index": 0.01 * (1 + year / 500),
                "pop_growth_rate": 0.001 * (1 + year / 5000),
                "urban_share_mean": 0.02,
            })
    return pd.DataFrame(rows)

def test_build_malthusian_panel(wide_panel):
    result = build_malthusian_panel(wide_panel, entity_col="country")
    assert "popdens_lag" in result.columns
    assert "land_labor_ratio_lag" in result.columns
    assert "intens_x_density" in result.columns
    assert len(result) < len(wide_panel)

def test_malthusian_panel_no_future_leakage(wide_panel):
    result = build_malthusian_panel(wide_panel, entity_col="country")
    fra = result[result["country"] == "FRA"].sort_values("year")
    for i in range(1, len(fra)):
        assert fra.iloc[i]["popdens_lag"] == pytest.approx(fra.iloc[i - 1]["popdens_p_km2_mean"])
