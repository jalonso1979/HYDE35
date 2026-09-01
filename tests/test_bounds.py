import pytest
import pandas as pd
import numpy as np

from analysis.paper2_malthus.bounds import estimate_bounds, bounds_summary_table

def test_estimate_bounds():
    np.random.seed(42)
    countries = ["A", "B", "C"]
    years = [1000, 1100, 1200, 1300]

    data1 = []
    data2 = []
    for c in countries:
        for y in years:
            data1.append({
                "country": c,
                "year": y,
                "pop_growth_rate": np.random.randn(),
                "popdens_lag": np.random.randn(),
                "land_labor_ratio_lag": np.random.randn()
            })
            data2.append({
                "country": c,
                "year": y,
                "pop_growth_rate": np.random.randn(),
                "popdens_lag": np.random.randn() * 2,
                "land_labor_ratio_lag": np.random.randn() * 0.5
            })

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    scenario_panels = {
        "baseline": df1,
        "high": df2
    }

    res = estimate_bounds(
        scenario_panels,
        dep_var="pop_growth_rate",
        indep_vars=["popdens_lag", "land_labor_ratio_lag"],
        key_var="popdens_lag",
        entity_col="country"
    )

    assert "bounds" in res
    assert "scenario_results" in res

    lb, ub = res["bounds"]
    assert isinstance(lb, float)
    assert isinstance(ub, float)
    assert lb <= ub

    assert set(res["scenario_results"].keys()) == {"baseline", "high"}
    for label in ["baseline", "high"]:
        res_dict = res["scenario_results"][label]
        assert "coef" in res_dict
        assert "pvalue" in res_dict
        assert "nobs" in res_dict
        assert res_dict["nobs"] == len(countries) * len(years)

def test_estimate_bounds_default_indep_vars():
    np.random.seed(42)
    countries = ["A", "B", "C"]
    years = [1000, 1100, 1200, 1300]

    data = []
    for c in countries:
        for y in years:
            data.append({
                "country": c,
                "year": y,
                "pop_growth_rate": np.random.randn(),
                "popdens_lag": np.random.randn(),
                "land_labor_ratio_lag": np.random.randn()
            })

    df1 = pd.DataFrame(data)

    scenario_panels = {
        "baseline": df1,
    }

    res = estimate_bounds(
        scenario_panels,
        # dep_var, indep_vars, key_var, entity_col left to defaults
    )

    assert "bounds" in res
    assert "baseline" in res["scenario_results"]

def test_bounds_summary_table():
    bounds_result = {
        "bounds": (-0.5, 0.3),
        "scenario_results": {
            "baseline": {"coef": -0.2, "pvalue": 0.05, "nobs": 100},
            "high": {"coef": 0.1, "pvalue": 0.1, "nobs": 100}
        }
    }

    df = bounds_summary_table(bounds_result)

    assert len(df) == 3
    assert df.iloc[0]["scenario"] == "baseline"
    assert df.iloc[1]["scenario"] == "high"
    assert df.iloc[2]["scenario"] == "BOUNDS"

    assert df.iloc[2]["coef"] == "[-0.500000, 0.300000]"
    assert pd.isna(df.iloc[2]["pvalue"])
    assert pd.isna(df.iloc[2]["nobs"])
