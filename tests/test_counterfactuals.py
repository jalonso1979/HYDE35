import numpy as np
import pandas as pd
import pytest
from analysis.paper3_climate.counterfactuals import simulate_counterfactual, run_counterfactual_experiment

def test_simulate_counterfactual_identical_shocks():
    irf = np.array([1.0, 0.5, 0.25])
    actual = np.array([0.0, 0.0, 0.0, 0.0])
    cf = np.array([0.0, 0.0, 0.0, 0.0])
    baseline = np.array([10.0, 10.0, 10.0, 10.0])

    result = simulate_counterfactual(irf, actual, cf, baseline)
    np.testing.assert_allclose(result, baseline)

def test_simulate_counterfactual_single_shock():
    irf = np.array([1.0, 0.5, 0.25])
    actual = np.array([0.0, 0.0, 0.0, 0.0])
    cf = np.array([0.0, 1.0, 0.0, 0.0])
    baseline = np.array([10.0, 10.0, 10.0, 10.0])

    result = simulate_counterfactual(irf, actual, cf, baseline)
    expected = np.array([
        10.0,                 # t=0
        10.0 + 1.0,           # t=1: irf[0]*1
        10.0 + 0.5,           # t=2: irf[1]*1
        10.0 + 0.25,          # t=3: irf[2]*1
    ])
    np.testing.assert_allclose(result, expected)

def test_simulate_counterfactual_multiple_shocks():
    irf = np.array([2.0, 1.0])
    actual = np.array([0.0, 1.0, 0.0])
    cf = np.array([1.0, 0.0, 1.0])
    baseline = np.array([0.0, 0.0, 0.0])

    result = simulate_counterfactual(irf, actual, cf, baseline)
    expected = np.array([2.0, -1.0, 1.0])
    np.testing.assert_allclose(result, expected)

def test_run_counterfactual_experiment():
    irf_df = pd.DataFrame({
        "horizon": [0, 1],
        "coefficient": [2.0, 1.0]
    })

    climate_panel = pd.DataFrame({
        "region": ["A", "A", "A", "B", "B", "B"],
        "year": [2000, 2001, 2002, 2000, 2001, 2002],
        "temp_shock": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    })

    response_panel = pd.DataFrame({
        "region": ["A", "A", "A", "B", "B", "B"],
        "year": [2000, 2001, 2002, 2000, 2001, 2002],
        "gdp": [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]
    })

    result_df = run_counterfactual_experiment(
        irf_df=irf_df,
        climate_panel=climate_panel,
        response_panel=response_panel,
        source_entity="B",
        target_entity="A",
        shock_var="temp_shock",
        response_var="gdp"
    )

    assert list(result_df["year"]) == [2000, 2001, 2002]
    assert list(result_df["actual"]) == [10.0, 10.0, 10.0]
    np.testing.assert_allclose(result_df["counterfactual"], [12.0, 9.0, 11.0])
