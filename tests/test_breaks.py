import numpy as np
import pandas as pd
import pytest

from analysis.paper2_malthus.breaks import chow_test, detect_structural_breaks, detect_breaks_all_entities

def test_chow_test_no_break():
    # Generate data with NO structural break
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    beta = np.array([1.5, -2.0])
    y = X @ beta + np.random.randn(n) * 0.1

    # Test at midpoint
    f_stat, p_val = chow_test(y, X, n // 2)

    # Since there's no break, p-value should be large (fail to reject null)
    assert p_val > 0.05
    assert f_stat < 3.0 # F-stat should be relatively small

def test_chow_test_with_break():
    # Generate data WITH a structural break
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)

    # First half uses beta1, second half uses beta2
    beta1 = np.array([1.5, -2.0])
    beta2 = np.array([-1.5, 2.0])

    y = np.zeros(n)
    y[:50] = X[:50] @ beta1 + np.random.randn(50) * 0.1
    y[50:] = X[50:] @ beta2 + np.random.randn(50) * 0.1

    # Test at midpoint (actual break point)
    f_stat, p_val = chow_test(y, X, 50)

    # Since there's a strong break, p-value should be very small
    assert p_val < 0.05
    assert f_stat > 10.0 # F-stat should be large

def test_detect_structural_breaks_too_short():
    # Test that early return triggers if dataset is too short
    df = pd.DataFrame({
        "country": ["A", "A", "A", "A"], # length 4
        "year": [1, 2, 3, 4],
        "pop_growth_rate": [0.1, 0.2, 0.1, 0.2],
        "popdens_lag": [1.0, 2.0, 3.0, 4.0]
    })
    # Min segment 3 requires at least 6 points
    breaks = detect_structural_breaks(df, entity="A", min_segment=3)
    assert len(breaks) == 0
    assert breaks == []

def test_detect_structural_breaks_with_break():
    # Test that structural break is correctly detected
    np.random.seed(42)
    n = 20
    years = np.arange(2000, 2000 + n)
    X_vals = np.random.randn(n)

    # Introduce break at index 10
    beta1, beta2 = 1.0, -1.0
    y_vals = np.zeros(n)
    y_vals[:10] = beta1 * X_vals[:10] + np.random.randn(10) * 0.05
    y_vals[10:] = beta2 * X_vals[10:] + np.random.randn(10) * 0.05

    df = pd.DataFrame({
        "country": ["A"] * n,
        "year": years,
        "pop_growth_rate": y_vals,
        "popdens_lag": X_vals
    })

    breaks = detect_structural_breaks(df, entity="A", min_segment=5)

    # Assert that at least one break is found
    assert len(breaks) > 0

    # The actual break is at 2010. Chow test might find significance nearby.
    # Check if the break is reasonably close to 2010
    break_years = [b["break_year"] for b in breaks]
    assert any(2008 <= yr <= 2012 for yr in break_years)

    for b in breaks:
        assert "break_year" in b
        assert "f_stat" in b
        assert "p_value" in b
        assert b["p_value"] < 0.05

def test_detect_breaks_all_entities_none_found():
    # Test when no entities have breaks
    np.random.seed(42)
    n = 20
    df = pd.DataFrame({
        "country": ["A"] * (n//2) + ["B"] * (n//2),
        "year": list(np.arange(2000, 2000 + n//2)) * 2,
        "pop_growth_rate": np.random.randn(n), # Just noise
        "popdens_lag": np.random.randn(n)
    })

    # Use very small alpha so nothing is significant
    res = detect_breaks_all_entities(df, min_segment=3, significance=1e-10)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert list(res.columns) == ["country", "break_year", "f_stat", "p_value"]

def test_detect_breaks_all_entities_found():
    # Test with one entity having a break
    np.random.seed(42)
    n_per_country = 20

    # Country A with break
    X_A = np.random.randn(n_per_country)
    y_A = np.zeros(n_per_country)
    y_A[:10] = 2.0 * X_A[:10] + np.random.randn(10) * 0.1
    y_A[10:] = -2.0 * X_A[10:] + np.random.randn(10) * 0.1

    # Country B with no break (just noise)
    X_B = np.random.randn(n_per_country)
    y_B = np.random.randn(n_per_country)

    df = pd.DataFrame({
        "country": ["A"] * n_per_country + ["B"] * n_per_country,
        "year": list(np.arange(2000, 2000 + n_per_country)) * 2,
        "pop_growth_rate": np.concatenate([y_A, y_B]),
        "popdens_lag": np.concatenate([X_A, X_B])
    })

    res = detect_breaks_all_entities(df, min_segment=5)

    assert isinstance(res, pd.DataFrame)
    assert len(res) > 0
    # Breaks should primarily come from A
    assert "A" in res["country"].values
