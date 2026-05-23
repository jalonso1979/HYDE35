"""Build a country-year ERA5 panel for the 7 Long Shadow countries.

All 7 countries map to ERA5 region 11 (Europe tile, lat 27-80N, lon 25W-31E).
For each country we average grid points within its bbox to get country-year
means of t2m + tp.

Output: era5_country_annual.parquet (columns: iso3, year, T, P)
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import xarray as xr

from analysis.paper4_shadow.long_shadow_fertility.data.era5_region_country_map import (
    COUNTRY_BBOXES,
    build_region_country_map,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ERA5_ROOT = Path("/Volumes/BIGDATA/HYDE35/ERA5")
OUT_PATH = Path(
    "/Volumes/BIGDATA/HYDE35/analysis/data/long_shadow_fertility/"
    "era5_country_annual.parquet"
)


def _load_region_files(region_id: int) -> list[Path]:
    """All extracted NetCDFs for a given region, sorted by year/month."""
    reg_dir = ERA5_ROOT / f"region={region_id}"
    if not reg_dir.exists():
        return []
    files: list[Path] = []
    for yr_dir in sorted(reg_dir.glob("year=*")):
        ext_dir = yr_dir / "_extracted"
        if not ext_dir.exists():
            continue
        files.extend(sorted(ext_dir.glob("*.nc")))
    return files


def _ds_var_names(ds: xr.Dataset) -> tuple[str | None, str | None]:
    """Return (t2m_var_name, tp_var_name) or (None, None) if not found."""
    t2m: str | None = None
    tp: str | None = None
    for v in ds.data_vars:
        v_low = str(v).lower()
        if v_low in ("t2m", "2t", "temperature"):
            t2m = str(v)
        elif v_low in ("tp", "total_precipitation"):
            tp = str(v)
    return t2m, tp


def _coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lat = "latitude" if "latitude" in ds.coords else "lat"
    lon = "longitude" if "longitude" in ds.coords else "lon"
    return lat, lon


def _time_dim_name(ds: xr.Dataset) -> str | None:
    for cand in ("valid_time", "time", "month"):
        if cand in ds.dims:
            return cand
    return None


def _aggregate_country_year(
    ds: xr.Dataset, iso: str, bbox: tuple[float, float, float, float]
) -> dict[int, dict[str, float]]:
    """Slice ds by country bbox; return per-year accumulators (sum, n) for T and P."""
    lat_name, lon_name = _coord_names(ds)
    t2m_v, tp_v = _ds_var_names(ds)
    if t2m_v is None and tp_v is None:
        return {}

    lat_min, lat_max, lon_min, lon_max = bbox

    # ERA5 latitude often runs high-to-low; xarray slice() needs the order
    # that matches the coord direction
    lats = ds[lat_name].values
    if lats.size >= 2 and lats[0] > lats[-1]:  # descending
        lat_sel = slice(lat_max, lat_min)
    else:
        lat_sel = slice(lat_min, lat_max)

    # Handle 0..360 longitude convention vs -180..180
    lons = ds[lon_name].values
    if lons.size == 0:
        return {}
    if lons.min() >= 0 and lon_min < 0:
        lon_min_a = lon_min + 360
        lon_max_a = lon_max + 360 if lon_max < 0 else lon_max
        lon_sel = slice(min(lon_min_a, lon_max_a), max(lon_min_a, lon_max_a))
    else:
        lon_sel = slice(lon_min, lon_max)

    try:
        sub = ds.sel({lat_name: lat_sel, lon_name: lon_sel})
    except Exception:
        return {}

    if sub[lat_name].size == 0 or sub[lon_name].size == 0:
        return {}

    # Average over space; group by year over time
    spatial_dims = [d for d in sub.dims
                    if d not in ("time", "valid_time", "month", "number", "expver")]
    if not spatial_dims:
        return {}
    sub_mean = sub.mean(dim=spatial_dims, skipna=True)

    time_dim = _time_dim_name(sub_mean)
    if time_dim is None:
        return {}
    times = pd.to_datetime(sub_mean[time_dim].values)
    try:
        years = times.year
    except AttributeError:
        years = pd.DatetimeIndex(times).year

    rows: dict[int, dict[str, float]] = {}
    # Pull arrays once to avoid per-step .isel which is slow
    t2m_arr = sub_mean[t2m_v].values if t2m_v is not None else None
    tp_arr = sub_mean[tp_v].values if tp_v is not None else None

    for i in range(len(years)):
        year = int(years[i])
        if year not in rows:
            rows[year] = {"T_sum": 0.0, "T_n": 0, "P_sum": 0.0, "P_n": 0}
        if t2m_arr is not None:
            v = float(t2m_arr[i])
            if not np.isnan(v):
                rows[year]["T_sum"] += v
                rows[year]["T_n"] += 1
        if tp_arr is not None:
            v = float(tp_arr[i])
            if not np.isnan(v):
                rows[year]["P_sum"] += v
                rows[year]["P_n"] += 1
    return rows


def build_era5_country_panel(write: bool = False, verbose: bool = False) -> pd.DataFrame:
    region_map = build_region_country_map()  # iso3 -> region_id

    # Per-country, per-year accumulators (sum + count of monthly grid means)
    accum: dict[str, dict[int, dict[str, float]]] = {iso: {} for iso in COUNTRY_BBOXES}

    # For efficiency, we load each region NetCDF once and process all matching
    # countries against that single dataset.
    region_ids = set(region_map.values()) - {-1}
    for r in sorted(region_ids):
        files = _load_region_files(r)
        if verbose:
            print(f"region={r}: {len(files)} NCs")
        countries_in_r = [iso for iso, rid in region_map.items() if rid == r]
        for idx, nc_path in enumerate(files):
            try:
                ds = xr.open_dataset(nc_path)
            except Exception as exc:
                if verbose:
                    print(f"  skip {nc_path.name}: {exc}")
                continue
            try:
                for iso in countries_in_r:
                    bbox = COUNTRY_BBOXES[iso]
                    year_rows = _aggregate_country_year(ds, iso, bbox)
                    for year, vals in year_rows.items():
                        if year not in accum[iso]:
                            accum[iso][year] = {
                                "T_sum": 0.0, "T_n": 0,
                                "P_sum": 0.0, "P_n": 0,
                            }
                        for k, v in vals.items():
                            accum[iso][year][k] += v
            finally:
                ds.close()
            if verbose and (idx + 1) % 50 == 0:
                print(f"  processed {idx + 1}/{len(files)} files in region {r}")

    rows = []
    for iso, by_year in accum.items():
        for year, agg in sorted(by_year.items()):
            T = agg["T_sum"] / agg["T_n"] if agg["T_n"] > 0 else np.nan
            P = agg["P_sum"] / agg["P_n"] if agg["P_n"] > 0 else np.nan
            rows.append({"iso3": iso, "year": year, "T": T, "P": P})

    df = pd.DataFrame(rows).sort_values(["iso3", "year"]).reset_index(drop=True)

    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    df = build_era5_country_panel(write=True, verbose=True)
    yr_min = df["year"].min() if len(df) else "NA"
    yr_max = df["year"].max() if len(df) else "NA"
    print(f"wrote {OUT_PATH}: {len(df)} rows, "
          f"{df['iso3'].nunique() if len(df) else 0} countries, "
          f"{yr_min}-{yr_max}")
    print(df.head(10))
    print(df.tail(10))
