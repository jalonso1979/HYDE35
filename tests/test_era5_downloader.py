import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from analysis.shared import era5_downloader

def test_load_region_bboxes_exception_handling(tmp_path, monkeypatch):
    """
    Test that if `xr.open_dataset` raises an exception for a .nc file,
    the loop continues and tries the next .nc file, eventually succeeding.
    """
    # Setup dummy directory structure
    era5_root = tmp_path / "ERA5"
    era5_root.mkdir()

    reg_dir = era5_root / "region=1"
    reg_dir.mkdir()

    yr_dir = reg_dir / "year=2000"
    yr_dir.mkdir()

    ext_path = yr_dir / "_extracted"
    ext_path.mkdir()

    # Create two .nc files.
    # Note: `ext_path.iterdir()` does not guarantee order, but in our test we
    # mock based on the filename to ensure one fails and the other succeeds.
    nc_file1 = ext_path / "a_corrupt.nc"
    nc_file1.touch()

    nc_file2 = ext_path / "b_valid.nc"
    nc_file2.touch()

    monkeypatch.setattr(era5_downloader, "ERA5_ROOT", era5_root)

    # Prepare a mock dataset for the valid file
    mock_ds = MagicMock()
    mock_ds.latitude.values.max.return_value = 10.0
    mock_ds.latitude.values.min.return_value = 0.0
    mock_ds.longitude.values.max.return_value = 20.0
    mock_ds.longitude.values.min.return_value = 10.0

    def mock_open_dataset(filename, engine=None):
        if "a_corrupt" in str(filename):
            raise Exception("Mocked exception for corrupt file")
        return mock_ds

    with patch("analysis.shared.era5_downloader.xr.open_dataset", side_effect=mock_open_dataset) as mock_open:
        bboxes = era5_downloader.load_region_bboxes()

    assert 1 in bboxes
    assert bboxes[1]["north"] == 10.0
    assert bboxes[1]["south"] == 0.0
    assert bboxes[1]["east"] == 20.0
    assert bboxes[1]["west"] == 10.0

    # Verify open_dataset was called twice, showing it continued after the exception
    assert mock_open.call_count == 2
    mock_ds.close.assert_called_once()
