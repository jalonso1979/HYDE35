"""Tests for multinomial logit pathway prediction (Exercise 1)."""

import numpy as np
import pandas as pd
import pytest

from analysis.paper4_shadow.pathway_prediction import (
    compute_marginal_effects,
    run_pathway_multinomial,
)


@pytest.fixture
def synthetic_pathway_climate():
    """60 countries across 3 pathways where seasonality clearly predicts pathway."""
    rng = np.random.default_rng(42)

    n_per = 20
    pathways = np.repeat([0, 1, 2], n_per)

    # Seasonality separates pathways with moderate overlap to avoid
    # perfect separation (which causes MNLogit convergence issues).
    seasonality = np.concatenate([
        rng.normal(0.8, 0.15, n_per),  # pathway 0 (intensive)
        rng.normal(0.4, 0.15, n_per),  # pathway 1 (mixed)
        rng.normal(0.1, 0.15, n_per),  # pathway 2 (pastoral)
    ])

    mean_temp = rng.normal(15, 5, 3 * n_per)
    latitude = rng.uniform(-60, 60, 3 * n_per)

    df = pd.DataFrame({
        "pathway": pathways,
        "seasonality": seasonality,
        "mean_temp": mean_temp,
        "latitude": latitude,
    })
    return df


def test_multinomial_runs(synthetic_pathway_climate):
    """Result has model, summary, predictions keys."""
    result = run_pathway_multinomial(synthetic_pathway_climate)
    assert "model" in result
    assert "summary" in result
    assert "predictions" in result


def test_multinomial_seasonality_significant(synthetic_pathway_climate):
    """At least one seasonality p-value < 0.10."""
    result = run_pathway_multinomial(synthetic_pathway_climate)
    pvals = result["pvalues"]
    # pvalues is a DataFrame; find seasonality rows
    season_pvals = pvals.loc["seasonality"]
    assert (season_pvals < 0.10).any(), (
        f"No seasonality p-value < 0.10: {season_pvals}"
    )


def test_marginal_effects_shape(synthetic_pathway_climate):
    """Correct number of rows: n_vars x (n_pathways - 1)."""
    result = run_pathway_multinomial(
        synthetic_pathway_climate,
        climate_vars=["seasonality"],
        control_vars=["mean_temp", "latitude"],
    )
    me = compute_marginal_effects(
        result["model"],
        synthetic_pathway_climate,
        climate_vars=["seasonality"],
    )
    n_vars = 1  # just seasonality
    n_pathways_minus1 = 2  # 3 pathways, base=0
    expected_rows = n_vars * n_pathways_minus1
    assert len(me) == expected_rows
    assert "variable" in me.columns
    assert "pathway" in me.columns
    assert "marginal_effect" in me.columns
