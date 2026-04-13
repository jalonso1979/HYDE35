"""Tests for escape mechanism interaction regressions (Exercise 5)."""

import numpy as np
import pandas as pd
import pytest

from analysis.paper4_shadow.escape_mechanism import (
    run_escape_interactions,
    run_rolling_with_mediator,
)


@pytest.fixture
def synthetic_modern_panel():
    """20 countries, 1950-2025, with known interaction structure.

    DGP: growth = 0.02 - 0.005*shock + 0.004*shock*urban + noise
    so urban_share interaction coef should be positive (reduces harm).
    """
    rng = np.random.default_rng(42)
    countries = 20
    years = np.arange(1950, 2026)
    rows = []
    for c in range(countries):
        for y in years:
            t = y - 1950
            urban = np.clip(0.1 + 0.005 * t + rng.normal(0, 0.02), 0, 1)
            intens = np.clip(0.3 + 0.003 * t + rng.normal(0, 0.02), 0, 1)
            shock = rng.normal(0, 1)
            growth = (
                0.02
                - 0.005 * shock
                + 0.004 * shock * urban
                + rng.normal(0, 0.002)
            )
            rows.append(
                {
                    "country_id": f"C{c:02d}",
                    "year": y,
                    "urban_share": urban,
                    "intensification_index": intens,
                    "temp_anomaly": shock,
                    "pop_growth": growth,
                }
            )
    return pd.DataFrame(rows)


def test_escape_interactions_runs(synthetic_modern_panel):
    """run_escape_interactions returns 3 results: urban, intens, joint."""
    results = run_escape_interactions(synthetic_modern_panel)
    assert "urban_share" in results
    assert "intensification_index" in results
    assert "joint" in results
    assert len(results) == 3


def test_escape_interaction_coefficients(synthetic_modern_panel):
    """Urban interaction coefficient should be positive (reduces harm)."""
    results = run_escape_interactions(synthetic_modern_panel)
    urban_result = results["urban_share"]
    assert urban_result is not None
    assert urban_result["interaction_coef"] > 0, (
        f"Urban interaction coef should be > 0, got {urban_result['interaction_coef']:.6f}"
    )


def test_rolling_with_mediator(synthetic_modern_panel):
    """Rolling estimation returns non-empty DataFrame with expected columns."""
    df = run_rolling_with_mediator(
        synthetic_modern_panel,
        shock_var="temp_anomaly",
        outcome_var="pop_growth",
        mediator_var="urban_share",
        entity_col="country_id",
        window=20,
        step=5,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in ["center_year", "shock_coef", "interaction_coef", "interaction_pval", "nobs"]:
        assert col in df.columns, f"Missing column: {col}"
