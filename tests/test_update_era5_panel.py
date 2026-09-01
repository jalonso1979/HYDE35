import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
import runpy

def test_era5_error_handling(capsys):
    """
    Test that when xr.open_dataset throws an exception for t2m and tp,
    the fallback nan values are correctly assigned, and the script doesn't crash.
    """
    # Create mock paths
    reg_dir = Path("/mock/region=1")
    yr_dir = Path("/mock/region=1/year=2020")
    ext_dir = Path("/mock/region=1/year=2020/_extracted")

    t2m_file = ext_dir / "era5_1_2020.m0.nc"
    tp_file = ext_dir / "era5_1_2020.m1.nc"

    # Mock glob to simulate filesystem structure
    def mock_glob_impl(self, pattern):
        if pattern == "region=*":
            return [reg_dir]
        elif pattern == "year=*":
            return [yr_dir]
        elif pattern == "*.m0.nc":
            return [t2m_file]
        elif pattern == "*.m1.nc":
            return [tp_file]
        return []

    original_exists = Path.exists
    def mock_exists(self):
        # We need _extracted to exist, and instant/accum files to NOT exist
        # to trigger the .m0.nc / .m1.nc branch
        if str(self) == str(ext_dir):
            return True
        if "data_stream" in str(self):
            return False
        return original_exists(self)

    # Patch the glob and exists methods on pathlib.Path
    # Also patch DataFrame.to_parquet so we don't try to write to a fake volume
    # And patch xarray.open_dataset to raise an exception, simulating a bad/corrupt file
    with patch.object(Path, 'glob', autospec=True) as mock_glob, \
         patch.object(Path, 'exists', autospec=True) as mock_exists_patch, \
         patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet, \
         patch("xarray.open_dataset", side_effect=Exception("Mocked xarray error")):

        mock_glob.side_effect = mock_glob_impl
        mock_exists_patch.side_effect = mock_exists

        # Run the script
        run_dict = runpy.run_path("analysis/update_era5_panel.py")

        # Extract the resulting DataFrame
        era5_panel = run_dict["era5_panel"]

        # Verify that we processed exactly 1 row
        assert len(era5_panel) == 1

        row = era5_panel.iloc[0]
        assert row["region"] == 1
        assert row["year"] == 2020

        # Verify that temperature values are NaN due to the exception
        assert pd.isna(row["temperature_k"])
        assert pd.isna(row["temperature_c"])

        # Verify that precipitation values are NaN due to the exception
        assert pd.isna(row["precipitation_m"])
        assert pd.isna(row["precipitation_mm"])

        # Verify warnings were printed
        captured = capsys.readouterr()
        assert "WARN: region=1 year=2020 t2m failed: Mocked xarray error" in captured.out
        assert "WARN: region=1 year=2020 tp failed: Mocked xarray error" in captured.out


def test_era5_missing_tp_handling(capsys):
    """
    Test that when precipitation files are completely missing, the tp values are NaN
    but t2m is calculated correctly.
    """
    reg_dir = Path("/mock/region=2")
    yr_dir = Path("/mock/region=2/year=2021")
    ext_dir = Path("/mock/region=2/year=2021/_extracted")

    t2m_file = ext_dir / "era5_2_2021.m0.nc"

    def mock_glob_impl(self, pattern):
        if pattern == "region=*":
            return [reg_dir]
        elif pattern == "year=*":
            return [yr_dir]
        elif pattern == "*.m0.nc":
            return [t2m_file]
        elif pattern == "*.m1.nc":
            return []  # Missing tp files
        return []

    original_exists = Path.exists
    def mock_exists(self):
        if str(self) == str(ext_dir):
            return True
        if "data_stream" in str(self):
            return False
        return original_exists(self)

    # For the t2m file, we need a mock dataset
    class MockDataset:
        def __init__(self, var_name, value):
            class MockDataArray:
                def __init__(self, val):
                    self.val = val
                def mean(self):
                    return self
                @property
                def values(self):
                    return self.val
            self.data = {var_name: MockDataArray(value)}

        def __getitem__(self, key):
            return self.data[key]

        def close(self):
            pass

    def mock_open_dataset_impl(filepath, **kwargs):
        if "m0.nc" in str(filepath):
            return MockDataset("t2m", 300.0)  # 300 K
        raise FileNotFoundError(f"File not found: {filepath}")

    with patch.object(Path, 'glob', autospec=True) as mock_glob, \
         patch.object(Path, 'exists', autospec=True) as mock_exists_patch, \
         patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet, \
         patch("xarray.open_dataset", side_effect=mock_open_dataset_impl):

        mock_glob.side_effect = mock_glob_impl
        mock_exists_patch.side_effect = mock_exists

        run_dict = runpy.run_path("analysis/update_era5_panel.py")
        era5_panel = run_dict["era5_panel"]

        assert len(era5_panel) == 1
        row = era5_panel.iloc[0]

        # Check temperature calculation (300K -> 26.85C)
        assert row["temperature_k"] == 300.0
        assert np.isclose(row["temperature_c"], 300.0 - 273.15)

        # Check precipitation fallback to NaN
        assert pd.isna(row["precipitation_m"])
        assert pd.isna(row["precipitation_mm"])
