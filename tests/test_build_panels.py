import pandas as pd
import numpy as np
import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock config BEFORE importing analysis modules to avoid hardcoded volume paths
class MockConfig:
    DATA_ROOT = MagicMock()
    SCENARIOS = {}
    SCENARIO_LABELS = []
    ANALYSIS_DATA = MagicMock()
    ANALYSIS_DATA.mkdir = MagicMock()
    COUNTRY_YEAR_CSV = "mock_country.csv"
    REGION_YEAR_CSV = "mock_region.csv"
    ANTHRO_REGION_CSV = "mock_anthro.csv"
    GRAZING_CALORIC_WEIGHT = 0.5

sys.modules['analysis.shared.config'] = MockConfig

from analysis.shared.build_panels import (
    build_country_analysis_panel,
    build_region_analysis_panel,
    save_panels
)

@pytest.fixture
def mock_country_panel_data():
    """Mock country-level panel data in long format."""
    rows = []
    for year in [1900, 1910]:
        for country in ["USA", "FRA"]:
            for var in ["nonrice_mha", "rice_mha", "grazing_mha", "pop_persons", "irrigation_share"]:
                rows.append({
                    "year": year,
                    "country": country,
                    "var": var,
                    "units": "some_unit",
                    "mean": 10.0 if var != "pop_persons" else 1000.0,
                    "std": 1.0
                })
    return pd.DataFrame(rows)

@pytest.fixture
def mock_region_panel_data():
    """Mock region-level scenario data in long format."""
    rows = []
    for scenario in ["base", "upper"]:
        for year in [1900, 1910]:
            for region in ["EU", "NA"]:
                for var in ["nonrice_cropland_mha", "rice_mha", "grazing_mha", "pop"]:
                    rows.append({
                        "scenario": scenario,
                        "year": year,
                        "region": region,
                        "var": var,
                        "units": "some_unit",
                        "value": 15.0 if var != "pop" else 2000.0
                    })
    return pd.DataFrame(rows)

def test_build_country_analysis_panel(mock_country_panel_data):
    """Test building country-level analysis panel."""
    with patch("analysis.shared.build_panels.load_existing_country_panel", return_value=mock_country_panel_data):
        df = build_country_analysis_panel()

        # Check expected columns
        expected_cols = [
            "country", "year",
            "nonrice_mha_mean", "rice_mha_mean", "grazing_mha_mean", "pop_persons_mean", "irrigation_share_mean",
            "nonrice_mha_std", "rice_mha_std", "grazing_mha_std", "pop_persons_std", "irrigation_share_std",
            "land_labor_ratio", "ag_output_proxy_mha", "intensification_index", "pop_growth_rate"
        ]
        for col in expected_cols:
            assert col in df.columns

        assert len(df) == 4

        # Check population growth rate logic
        # Year 1900 should be NaN since there is no previous year
        assert df.loc[df["year"] == 1900, "pop_growth_rate"].isna().all()
        # Year 1910 should be 0.0 since population is constant (1000.0 for both years)
        assert (df.loc[df["year"] == 1910, "pop_growth_rate"] == 0.0).all()

        # Check specific computation results
        # land_labor_ratio: (cropland + grazing) / population = (10 + 10 + 10) / 1000 = 0.03
        assert np.isclose(df["land_labor_ratio"].iloc[0], 0.03)

        # ag_output_proxy_mha: cropland + grazing_weight * grazing = (10 + 10) + 0.5 * 10 = 25.0
        assert np.isclose(df["ag_output_proxy_mha"].iloc[0], 25.0)

def test_build_region_analysis_panel(mock_region_panel_data):
    """Test building region-level analysis panel."""
    with patch("analysis.shared.build_panels.load_existing_scenario_panel", return_value=mock_region_panel_data):
        df = build_region_analysis_panel()

        # Check expected columns (including uncertainty stats: mean, std, se)
        expected_cols = [
            "region", "year",
            "nonrice_cropland_mha_mean", "rice_mha_mean", "grazing_mha_mean", "pop_mean",
            "nonrice_cropland_mha_std", "rice_mha_std", "grazing_mha_std", "pop_std",
            "nonrice_cropland_mha_se", "rice_mha_se", "grazing_mha_se", "pop_se",
            "land_labor_ratio", "ag_output_proxy_mha", "pop_growth_rate"
        ]
        for col in expected_cols:
            assert col in df.columns

        assert len(df) == 4

        # Check population growth rate logic
        assert df.loc[df["year"] == 1900, "pop_growth_rate"].isna().all()
        assert (df.loc[df["year"] == 1910, "pop_growth_rate"] == 0.0).all()

        # Check specific computation results
        # land_labor_ratio: (cropland + grazing) / population = (15 + 15 + 15) / 2000 = 45 / 2000 = 0.0225
        assert np.isclose(df["land_labor_ratio"].iloc[0], 0.0225)
        # ag_output_proxy_mha: cropland + grazing_weight * grazing = (15 + 15) + 0.5 * 15 = 37.5
        assert np.isclose(df["ag_output_proxy_mha"].iloc[0], 37.5)

@patch("analysis.shared.build_panels.build_country_analysis_panel")
@patch("analysis.shared.build_panels.build_region_analysis_panel")
@patch("analysis.shared.build_panels.ANALYSIS_DATA")
def test_save_panels(mock_analysis_data, mock_build_region, mock_build_country):
    """Test saving all panels to Parquet format."""
    # Set up mocks for DataFrames
    mock_country_df = MagicMock()
    mock_country_df.__len__.return_value = 10
    mock_build_country.return_value = mock_country_df

    mock_region_df = MagicMock()
    mock_region_df.__len__.return_value = 5
    mock_build_region.return_value = mock_region_df

    # Set up mock for file path creation
    mock_analysis_data.__truediv__.return_value = MagicMock()

    save_panels()

    # Ensure to_parquet was called for both panels
    mock_country_df.to_parquet.assert_called_once()
    mock_region_df.to_parquet.assert_called_once()
