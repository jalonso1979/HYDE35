"""Tests for Malthusian regressions."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper2_malthus.regressions import run_fe_regression, run_rolling_window

@pytest.fixture
def malthus_panel():
    rng = np.random.default_rng(42)
    rows = []
    for country in ["FRA", "GBR", "CHN", "IND", "EGY"]:
        for year in range(1000, 1800, 100):
            popdens = 10 + year / 100 + rng.normal(0, 2)
            rows.append({
                "year": year, "country": country,
                "pop_growth_rate": 0.01 - 0.0005 * popdens + rng.normal(0, 0.001),
                "popdens_lag": popdens,
                "land_labor_ratio_lag": 0.02 - popdens * 0.0001,
                "intens_x_density": 0.01 * popdens,
                "log_popdens_lag": np.log(max(popdens, 0.1)),
                "log_land_labor_lag": np.log(max(0.02 - popdens * 0.0001, 1e-10)),
                "intensification_index": 0.01 + year / 10000,
            })
    return pd.DataFrame(rows)

def test_run_fe_regression(malthus_panel):
    result = run_fe_regression(
        malthus_panel, dep_var="pop_growth_rate",
        indep_vars=["popdens_lag", "land_labor_ratio_lag"], entity_col="country",
    )
    assert "params" in result
    assert "pvalues" in result
    assert "popdens_lag" in result["params"]
    assert result["params"]["popdens_lag"] < 0

def test_run_rolling_window(malthus_panel):
    results = run_rolling_window(
        malthus_panel, dep_var="pop_growth_rate", key_var="popdens_lag",
        control_vars=["land_labor_ratio_lag"], entity_col="country",
        window_years=400, step_years=100,
    )
    assert len(results) > 0
    assert "center_year" in results.columns
    assert "coefficient" in results.columns
    assert "pvalue" in results.columns
