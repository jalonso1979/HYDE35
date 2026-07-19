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


def test_grouped_substrate_identity():
    """Coalition substrate: sum of Shapley values still equals R²(full) - R²(baseline)."""
    rng = np.random.default_rng(7)
    n = 300
    X = rng.standard_normal((n, 5))
    y = X[:, 0] * 0.4 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + X[:, 3] * 0.1 + rng.standard_normal(n) * 0.4
    df = pd.DataFrame(X, columns=["a1", "a2", "b", "c", "d"])
    df["y"] = y
    # Bundle (a1, a2) as one coalition player; b, c, d are individual.
    result = shapley_r2_decomposition(
        df, y_col="y",
        substrates=[("a1", "a2"), "b", "c", "d"],
        controls=[],
    )
    keys = list(result["shapley"].keys())
    assert keys == ["a1+a2", "b", "c", "d"]
    s = result["shapley"]
    full_r2 = result["full_model_r2"]
    baseline_r2 = result["baseline_r2"]
    assert abs(sum(s.values()) - (full_r2 - baseline_r2)) < 1e-6


def test_grouped_substrate_matches_one_member_when_others_orthogonal():
    """When a2 is pure noise, the bundle (a1, a2) Shapley should equal Shapley of a1 alone."""
    rng = np.random.default_rng(11)
    n = 800
    X = rng.standard_normal((n, 4))
    # a2 is uncorrelated with y; bundle Shapley should be dominated by a1
    y = X[:, 0] * 1.0 + X[:, 2] * 0.5 + rng.standard_normal(n) * 0.3
    df = pd.DataFrame(X, columns=["a1", "a2", "b", "c"])
    df["y"] = y
    r_bundle = shapley_r2_decomposition(
        df, y_col="y",
        substrates=[("a1", "a2"), "b", "c"],
        controls=[],
    )
    r_alone = shapley_r2_decomposition(
        df, y_col="y",
        substrates=["a1", "b", "c"],
        controls=[],
    )
    # The bundle's Shapley should be close to a1's alone, within tolerance (a2
    # adds a small amount of overfit R² but is noise).
    bundle_phi = r_bundle["shapley"]["a1+a2"]
    a1_phi = r_alone["shapley"]["a1"]
    assert abs(bundle_phi - a1_phi) < 0.02


def test_backward_compatibility_with_strings():
    """Original 4-string substrate API still works."""
    rng = np.random.default_rng(3)
    n = 200
    X = rng.standard_normal((n, 4))
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.standard_normal(n) * 0.4
    df = pd.DataFrame(X, columns=["s1", "s2", "s3", "s4"])
    df["y"] = y
    result = shapley_r2_decomposition(
        df, y_col="y", substrates=["s1", "s2", "s3", "s4"], controls=[])
    s = result["shapley"]
    assert set(s.keys()) == {"s1", "s2", "s3", "s4"}
    assert abs(sum(s.values()) - (result["full_model_r2"] - result["baseline_r2"])) < 1e-6
