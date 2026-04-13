"""Tests for dual-channel separation regressions (Exercise 2)."""

import numpy as np
import pandas as pd
import pytest

from analysis.paper4_shadow.dual_channel import (
    run_dual_channel_malthusian,
    run_dual_channel_pathway,
)


@pytest.fixture
def synthetic_dual_data():
    """80 synthetic countries with known seasonality/volatility structure."""
    rng = np.random.default_rng(42)
    n = 80
    seasonality = rng.uniform(0.1, 1.0, size=n)
    volatility = rng.uniform(0.1, 1.0, size=n)

    # Pathway determined by seasonality thresholds
    pathway = np.where(
        seasonality > 0.6, 0, np.where(seasonality > 0.3, 1, 2)
    )

    # Malthusian beta driven by volatility + noise
    malthusian_beta = -0.001 * volatility + rng.normal(0, 0.0001, size=n)

    return pd.DataFrame(
        {
            "country": [f"C{i}" for i in range(n)],
            "seasonality": seasonality,
            "volatility": volatility,
            "pathway": pathway,
            "malthusian_beta": malthusian_beta,
        }
    )


def test_dual_channel_pathway_returns_results(synthetic_dual_data):
    """Pathway regression returns all expected keys."""
    result = run_dual_channel_pathway(synthetic_dual_data)
    for key in [
        "seasonality_coef",
        "seasonality_pval",
        "volatility_coef",
        "volatility_pval",
        "rsquared",
        "nobs",
        "model",
    ]:
        assert key in result, f"Missing key: {key}"
    assert result["nobs"] == 80


def test_dual_channel_malthusian_returns_results(synthetic_dual_data):
    """Malthusian regression returns all expected keys."""
    result = run_dual_channel_malthusian(synthetic_dual_data)
    for key in [
        "seasonality_coef",
        "seasonality_pval",
        "volatility_coef",
        "volatility_pval",
        "rsquared",
        "nobs",
        "model",
    ]:
        assert key in result, f"Missing key: {key}"
    assert result["nobs"] == 80


def test_dual_channel_orthogonality(synthetic_dual_data):
    """Seasonality should be a stronger predictor of pathway than volatility."""
    result = run_dual_channel_pathway(synthetic_dual_data)
    assert result["seasonality_pval"] < result["volatility_pval"], (
        f"Seasonality p-val ({result['seasonality_pval']:.4f}) should be "
        f"< volatility p-val ({result['volatility_pval']:.4f}) for pathway"
    )
