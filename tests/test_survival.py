"""Tests for time-to-Malthusian-exit survival analysis."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper1_escape.survival import (
    build_survival_dataset,
    fit_cox_model,
)


@pytest.fixture
def trajectory_features():
    return pd.DataFrame({
        "country": ["FRA", "GBR", "CHN", "BRA"],
        "peak_extensification_year": [1200, 1100, 800, 1500],
        "intensification_onset_year": [1400, 1300, 1000, np.nan],
        "urbanization_takeoff_year": [1500, 1400, 1200, np.nan],
        "max_pop_density": [50, 60, 120, 10],
        "min_land_labor_ratio": [0.005, 0.004, 0.002, 0.02],
    })


def test_build_survival_dataset(trajectory_features):
    result = build_survival_dataset(
        trajectory_features,
        event_col="intensification_onset_year",
        origin_col="peak_extensification_year",
        entity_col="country",
    )
    assert "duration" in result.columns
    assert "event" in result.columns
    fra = result[result["country"] == "FRA"].iloc[0]
    assert fra["duration"] == 200
    assert fra["event"] == 1
    bra = result[result["country"] == "BRA"].iloc[0]
    assert bra["event"] == 0


def test_fit_cox_model(trajectory_features):
    surv_df = build_survival_dataset(
        trajectory_features,
        event_col="intensification_onset_year",
        origin_col="peak_extensification_year",
        entity_col="country",
    )
    surv_df["max_pop_density"] = trajectory_features.set_index("country").loc[
        surv_df["country"].values, "max_pop_density"
    ].values
    summary = fit_cox_model(surv_df, covariates=["max_pop_density"])
    assert summary is not None
    assert "coef" in summary.columns
