"""Data loading functions for HYDE 3.5 NetCDF, ASCII grids, and CSVs."""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


def normalize_longitudes(lon: np.ndarray) -> np.ndarray:
    """Convert longitudes from 0..360 to -180..180 if needed, then sort."""
    lon = np.asarray(lon, dtype="float64")
    if lon.max() > 180.0:
        lon = np.where(lon > 180.0, lon - 360.0, lon)
    return np.sort(lon)


def read_esri_ascii_grid(path: Path) -> xr.DataArray:
    """Read an ESRI ASCII grid file into an xarray DataArray.

    Parses the 6-line header (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value)
    then reads the data block.
    """
    path = Path(path)
    header = {}
    with open(path) as f:
        for _ in range(6):
            line = f.readline().strip().split()
            header[line[0].lower()] = float(line[1])

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    xll = header["xllcorner"]
    yll = header["yllcorner"]
    cellsize = header["cellsize"]
    nodata = header["nodata_value"]

    data = np.loadtxt(path, skiprows=6, dtype="float64")
    data[data == nodata] = np.nan

    lon = np.arange(xll + cellsize / 2, xll + ncols * cellsize, cellsize)
    lat = np.arange(yll + (nrows - 0.5) * cellsize, yll, -cellsize)

    return xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": lat[:nrows], "lon": lon[:ncols]},
    )


def align_grid(da: xr.DataArray, template: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """Align a DataArray's lon/lat to match a template grid using nearest reindex."""
    t_lon = template["lon"].values if hasattr(template, "lon") else template.coords["lon"].values
    t_lat = template["lat"].values if hasattr(template, "lat") else template.coords["lat"].values

    # Normalize longitudes to match template convention
    da_lon = da["lon"].values.copy()
    if t_lon.min() < 0 and da_lon.min() >= 0:
        da_lon = normalize_longitudes(da_lon)
        da = da.assign_coords(lon=da_lon).sortby("lon")
    elif t_lon.min() >= 0 and da_lon.min() < 0:
        da_lon = np.where(da_lon < 0, da_lon + 360.0, da_lon)
        da = da.assign_coords(lon=da_lon).sortby("lon")

    return da.reindex(lat=t_lat, lon=t_lon, method="nearest")


def load_nc_variable(scenario_dir: Path, variable: str) -> xr.DataArray:
    """Load a single NetCDF variable from a scenario directory.

    Returns the first data variable found in the file as a DataArray
    with dimensions (time, lat, lon).
    """
    nc_path = scenario_dir / "NetCDF" / f"{variable}.nc"
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    # Get the first (and typically only) data variable
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]

    # Normalize longitudes to -180..180
    if da["lon"].values.max() > 180.0:
        new_lon = normalize_longitudes(da["lon"].values)
        da = da.assign_coords(lon=("lon", normalize_longitudes(da["lon"].values)))
        da = da.sortby("lon")

    return da


def load_existing_country_panel(path: Path) -> pd.DataFrame:
    """Load the pre-built country x year panel CSV."""
    df = pd.read_csv(path)
    expected_cols = ["year", "country", "var", "units", "mean", "std"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")
    return df


def load_existing_scenario_panel(path: Path) -> pd.DataFrame:
    """Load the pre-built scenario x region x year panel CSV."""
    df = pd.read_csv(path)
    expected_cols = ["scenario", "year", "region", "var", "units", "value"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")
    return df
