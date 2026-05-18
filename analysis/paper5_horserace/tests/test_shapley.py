"""Tests for Shapley R² decomposition."""
import numpy as np
import pandas as pd
import pytest

from analysis.paper5_horserace.shapley import shapley_r2_decomposition


def test_shapley_sums_to_total_r2():
    """Sum of Shapley values equals R² of full model minus R² of baseline."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 4))
    controls = rng.standard_normal((n, 2))
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + controls[:, 0] * 0.4 + rng.standard_normal(n) * 0.5
    df = pd.DataFrame(X, columns=["s1", "s2", "s3", "s4"])
    df[["c1", "c2"]] = controls
    df["y"] = y
    result = shapley_r2_decomposition(
        df, y_col="y", substrates=["s1", "s2", "s3", "s4"], controls=["c1", "c2"])

    full_r2 = result["full_model_r2"]
    baseline_r2 = result["baseline_r2"]
    shapley_sum = sum(result["shapley"].values())
    assert abs(shapley_sum - (full_r2 - baseline_r2)) < 1e-6


def test_shapley_correctly_orders_substrates():
    """A substrate with strong coefficient should have larger Shapley value."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 4))
    y = X[:, 0] * 1.0 + X[:, 1] * 0.1 + rng.standard_normal(n) * 0.5
    df = pd.DataFrame(X, columns=["s1", "s2", "s3", "s4"])
    df["y"] = y
    result = shapley_r2_decomposition(df, y_col="y",
                                       substrates=["s1", "s2", "s3", "s4"], controls=[])
    s = result["shapley"]
    assert s["s1"] > s["s2"] > s["s3"]
    assert s["s1"] > s["s4"]


def test_handles_missing_data():
    """Function should drop rows with NaN in y/substrates/controls."""
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, np.nan], "s1": [0.1, 0.2, 0.3, 0.4]})
    result = shapley_r2_decomposition(df, y_col="y", substrates=["s1"], controls=[])
    assert result["n_obs"] == 3
