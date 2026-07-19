"""ERA5 growing-season anomaly country-year panel (Phase 9 Pillar A1).

Re-aggregates region 11 ERA5 monthly NetCDFs to April-September means per
country (subset to per-country bbox from era5_region_country_map.COUNTRY_BBOXES),
converts K -> degrees C, subtracts 1961-1990 baseline mean to form anomalies.
Output schema parallels ModE-RA's country_climate_annual.parquet for splicing.

Notes
-----
Raw downloads are stored as one file per (region, year, month) under
``ERA5/region=R/year=YYYY/era5_R_YYYYMM.nc`` in two container formats: the
May-2026 CDS deliveries are ZIP archives holding two streams
(``data_stream-oper_stepType-instant.nc`` with t2m and
``data_stream-oper_stepType-accum.nc`` with tp), while the July-2026 bulk
completion wrote plain merged netCDF4 files carrying both variables in one
dataset. Both hold *hourly* fields for a single month; the zip-era months
through 1966/67 are 1.0-degree grids, everything later is 0.25-degree. For
each (iso3, year, month) we accumulate (sum, count) over all hourly spatial
means, then derive a monthly mean by division. Annual growing-season values
are produced only when all six months (Apr-Sep) are present.

We read the per-month files directly because the on-disk ``_extracted/``
cache for years >=1968 was overwritten month-by-month and only retains the
last month extracted, which would produce truncated/incorrect annual means.
"""
from __future__ import annotations
from pathlib import Path
import io
import re
import warnings
import zipfile
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
    "era5_country_annual_v2.parquet"
)

GROWING_MONTHS = (4, 5, 6, 7, 8, 9)
BASELINE_WINDOW = (1961, 1990)
KELVIN_OFFSET = 273.15

_MONTH_ZIP_RE = re.compile(r"era5_\d+_(\d{4})(\d{2})\.nc$")


def _iter_region_monthly_datasets(region_id: int,
                                  months: tuple[int, ...] | None = GROWING_MONTHS):
    """Yield (year, month, xarray.Dataset) for each monthly file.

    Handles both on-disk container formats: zip archives with two NetCDF
    members (instant t2m + accum tp), which are merged into a single Dataset,
    and plain merged netCDF4 files that already carry ``t2m`` and ``tp``
    together. ``months=None`` yields every month; the default restricts to
    the growing season.
    """
    reg_dir = ERA5_ROOT / f"region={region_id}"
    if not reg_dir.exists():
        return
    for yr_dir in sorted(reg_dir.glob("year=*")):
        for nc_zip in sorted(yr_dir.glob("era5_*.nc")):
            mat = _MONTH_ZIP_RE.search(nc_zip.name)
            if not mat:
                continue
            year = int(mat.group(1))
            month = int(mat.group(2))
            if months is not None and month not in months:
                continue
            # Leading-magic check: zipfile.is_zipfile() false-positives on
            # some HDF5 files (it scans the file tail for the EOCD signature).
            with open(nc_zip, "rb") as fh:
                is_zip = fh.read(4) == b"PK\x03\x04"
            if not is_zip:
                # July-2026 bulk completion: plain merged netCDF4, t2m + tp
                # in one dataset.
                try:
                    ds = xr.open_dataset(nc_zip).load()
                except Exception:
                    continue
                yield year, month, ds
                continue
            try:
                with zipfile.ZipFile(nc_zip) as zf:
                    members = zf.namelist()
                    datasets = []
                    for name in members:
                        with zf.open(name) as f:
                            buf = io.BytesIO(f.read())
                        try:
                            # .load() forces eager read so we can release the
                            # in-memory zip buffer immediately
                            ds = xr.open_dataset(buf).load()
                        except Exception:
                            continue
                        finally:
                            buf.close()
                        datasets.append(ds)
                if not datasets:
                    continue
                if len(datasets) == 1:
                    merged = datasets[0]
                else:
                    try:
                        merged = xr.merge(datasets, compat="override", join="outer")
                    except Exception:
                        for d in datasets:
                            yield year, month, d
                            d.close()
                        continue
                    finally:
                        for d in datasets:
                            d.close()
                yield year, month, merged
            except zipfile.BadZipFile:
                continue


def _var_names(ds: xr.Dataset) -> tuple[str | None, str | None]:
    t2m, tp = None, None
    for v in ds.data_vars:
        vl = str(v).lower()
        if vl in ("t2m", "2t"):
            t2m = str(v)
        elif vl in ("tp", "total_precipitation"):
            tp = str(v)
    return t2m, tp


def _coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lat = "latitude" if "latitude" in ds.coords else "lat"
    lon = "longitude" if "longitude" in ds.coords else "lon"
    return lat, lon


def _time_dim(ds: xr.Dataset) -> str | None:
    for cand in ("valid_time", "time", "month"):
        if cand in ds.dims:
            return cand
    return None


def _aggregate_country_growing_season(
    ds: xr.Dataset, bbox: tuple[float, float, float, float]
) -> dict[tuple[int, int], dict[str, tuple[float, int]]]:
    """Slice ds by bbox; return per (year, month) (sum, count) for T and P.

    Aggregation is done over space (spatial mean) and time (sum + count of
    hourly steps). Months outside GROWING_MONTHS are dropped.
    """
    lat_name, lon_name = _coord_names(ds)
    t2m_v, tp_v = _var_names(ds)
    if t2m_v is None and tp_v is None:
        return {}
    lat_min, lat_max, lon_min, lon_max = bbox

    lats = ds[lat_name].values
    if lats.size == 0:
        return {}
    lat_sel = slice(lat_max, lat_min) if lats[0] > lats[-1] else slice(lat_min, lat_max)

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

    spatial_dims = [d for d in sub.dims
                    if d not in ("time", "valid_time", "month", "number", "expver")]
    if not spatial_dims:
        return {}
    sub_mean = sub.mean(dim=spatial_dims, skipna=True)

    tdim = _time_dim(sub_mean)
    if tdim is None:
        return {}
    times_raw = sub_mean[tdim].values
    if times_raw.size == 0:
        return {}

    if times_raw.dtype.kind == "M":
        times = pd.to_datetime(times_raw)
        years = np.asarray(times.year)
        months = np.asarray(times.month)
    else:
        years = (times_raw // 12 + 1850).astype(int)
        months = (times_raw % 12 + 1).astype(int)

    grow_mask = np.isin(months, GROWING_MONTHS)
    if not grow_mask.any():
        return {}

    t_arr = sub_mean[t2m_v].values if t2m_v is not None else None
    p_arr = sub_mean[tp_v].values if tp_v is not None else None

    out: dict[tuple[int, int], dict[str, tuple[float, int]]] = {}
    # Iterate by unique (year, month) within file to vectorise summing
    pairs = set(zip(years[grow_mask].tolist(), months[grow_mask].tolist()))
    for y, m in pairs:
        sel = (years == y) & (months == m)
        rec: dict[str, tuple[float, int]] = {}
        if t_arr is not None:
            vals = t_arr[sel]
            finite = ~np.isnan(vals)
            rec["T"] = (float(vals[finite].sum()), int(finite.sum()))
        if p_arr is not None:
            vals = p_arr[sel]
            finite = ~np.isnan(vals)
            rec["P"] = (float(vals[finite].sum()), int(finite.sum()))
        out[(int(y), int(m))] = rec
    return out


_MEMO: dict[str, pd.DataFrame] = {}


def build_era5_country_annual_v2(write: bool = False) -> pd.DataFrame:
    # With the complete archive the raw sweep takes ~35 min; memoize within
    # the process so tests and the splice builder don't re-sweep region 11.
    if "df" in _MEMO:
        df = _MEMO["df"].copy()
        if write:
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(OUT_PATH, index=False)
        return df
    region_map = build_region_country_map()
    # accum[iso][(year, month)] = {"T": [sum, count], "P": [sum, count]}
    accum: dict[str, dict[tuple[int, int], dict[str, list[float]]]] = {
        iso: {} for iso in COUNTRY_BBOXES
    }

    region_ids = {r for r in region_map.values() if r > 0}
    for r in sorted(region_ids):
        countries_in_r = [iso for iso, rid in region_map.items() if rid == r]
        if not countries_in_r:
            continue
        for year, month, ds in _iter_region_monthly_datasets(r):
            try:
                for iso in countries_in_r:
                    bbox = COUNTRY_BBOXES[iso]
                    file_rows = _aggregate_country_growing_season(ds, bbox)
                    for key, rec in file_rows.items():
                        slot = accum[iso].setdefault(
                            key, {"T": [0.0, 0], "P": [0.0, 0]}
                        )
                        for var, (s, n) in rec.items():
                            slot[var][0] += s
                            slot[var][1] += n
            finally:
                ds.close()

    # Convert (sum, count) per (year, month) -> monthly means; require all 6
    # growing-season months present per year
    rows = []
    for iso, by_ym in accum.items():
        # Build per-year monthly mean dict
        years: dict[int, dict[int, dict[str, float]]] = {}
        for (y, m), slot in by_ym.items():
            months = years.setdefault(y, {})
            entry: dict[str, float] = {}
            t_sum, t_n = slot["T"]
            p_sum, p_n = slot["P"]
            if t_n > 0:
                entry["T"] = t_sum / t_n
            if p_n > 0:
                entry["P"] = p_sum / p_n
            months[m] = entry
        for y in sorted(years):
            months = years[y]
            t_vals = [months[m]["T"] for m in GROWING_MONTHS
                      if m in months and "T" in months[m]]
            p_vals = [months[m]["P"] for m in GROWING_MONTHS
                      if m in months and "P" in months[m]]
            if len(t_vals) != len(GROWING_MONTHS) or len(p_vals) != len(GROWING_MONTHS):
                continue
            T_K = float(np.mean(t_vals))
            P_mh = float(np.mean(p_vals))
            rows.append({
                "iso3": iso,
                "year": y,
                "t_grow_C_raw": T_K - KELVIN_OFFSET,
                "p_grow_raw": P_mh,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        if write:
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(OUT_PATH, index=False)
        return df

    base_lo, base_hi = BASELINE_WINDOW
    anom_parts = []
    for iso, sub in df.groupby("iso3"):
        baseline = sub.loc[sub["year"].between(base_lo, base_hi)]
        if len(baseline) == 0:
            t_base = float("nan")
            p_base = float("nan")
        else:
            t_base = float(baseline["t_grow_C_raw"].mean())
            p_base = float(baseline["p_grow_raw"].mean())
        sub = sub.copy()
        sub["t_growing_era5"] = sub["t_grow_C_raw"] - t_base
        sub["p_growing_era5"] = sub["p_grow_raw"] - p_base
        anom_parts.append(sub)
    df = pd.concat(anom_parts, ignore_index=True)
    df["source"] = "ERA5_apr_sep_anom_61-90"
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    _MEMO["df"] = df.copy()

    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    df = build_era5_country_annual_v2(write=True)
    print(f"wrote {OUT_PATH}: {len(df)} rows, "
          f"{df['iso3'].nunique()} countries, "
          f"{df['year'].min()}-{df['year'].max()}")
    print(df.head())
    print()
    print("Per-country baseline check (mean of t_growing_era5 in 1961-1990):")
    bs = df[df["year"].between(1961, 1990)]
    print(bs.groupby("iso3")["t_growing_era5"].agg(["mean", "std", "count"]))
