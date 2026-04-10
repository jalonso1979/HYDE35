"""Tests for transition trajectory feature extraction."""
import numpy as np
import pandas as pd
import pytest
from analysis.paper1_escape.trajectories import (
    detect_peak_extensification,
    detect_intensification_onset,
    detect_urbanization_takeoff,
    extract_trajectory_features,
)


def test_detect_peak_extensification():
    years = np.array([1000, 1100, 1200, 1300, 1400, 1500])
    land_area = np.array([10.0, 12.0, 15.0, 25.0, 28.0, 29.0])
    peak_year = detect_peak_extensification(years, land_area)
    assert peak_year == 1300


def test_detect_peak_extensification_monotonic_decline():
    years = np.array([1000, 1100, 1200])
    land_area = np.array([30.0, 20.0, 10.0])
    peak_year = detect_peak_extensification(years, land_area)
    assert peak_year == 1100


def test_detect_intensification_onset():
    years = np.array([1000, 1100, 1200, 1300, 1400])
    intens_index = np.array([0.01, 0.02, 0.05, 0.12, 0.18])
    land_growth = np.array([np.nan, 0.05, 0.03, 0.005, 0.002])
    onset = detect_intensification_onset(
        years, intens_index, land_growth,
        intens_threshold=0.05, land_growth_threshold=0.01,
    )
    assert onset == 1300


def test_detect_intensification_onset_never():
    years = np.array([1000, 1100, 1200])
    intens_index = np.array([0.01, 0.02, 0.03])
    land_growth = np.array([np.nan, 0.05, 0.04])
    onset = detect_intensification_onset(years, intens_index, land_growth)
    assert np.isnan(onset)


def test_detect_urbanization_takeoff():
    years = np.array([1000, 1100, 1200, 1300, 1400])
    urban_share = np.array([0.01, 0.02, 0.03, 0.06, 0.12])
    takeoff = detect_urbanization_takeoff(years, urban_share, threshold=0.05)
    assert takeoff == 1300


def test_extract_trajectory_features():
    df = pd.DataFrame({
        "year": [1000, 1100, 1200, 1300, 1400, 1500],
        "country": "FRA",
        "pop_persons_mean": [1e6, 1.2e6, 1.5e6, 2.5e6, 3.0e6, 3.2e6],
        "popdens_p_km2_mean": [10, 12, 15, 25, 30, 32],
        "land_labor_ratio": [0.015, 0.013, 0.010, 0.006, 0.005, 0.005],
        "ag_output_proxy_mha": [10, 12, 15, 20, 21, 21.5],
        "intensification_index": [0.01, 0.02, 0.05, 0.12, 0.18, 0.22],
        "urban_share_mean": [0.01, 0.02, 0.03, 0.06, 0.12, 0.18],
    })
    features = extract_trajectory_features(df, entity_col="country")
    assert len(features) == 1
    row = features.iloc[0]
    assert row["country"] == "FRA"
    assert "peak_extensification_year" in features.columns
    assert "intensification_onset_year" in features.columns
    assert "urbanization_takeoff_year" in features.columns
