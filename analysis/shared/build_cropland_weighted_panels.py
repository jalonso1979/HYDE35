"""Cropland+grazing-weighted ModE-RA aggregation.

The user-facing question: in a country with a small habitable strip
(Egypt, Saudi Arabia, Australia interior, Russia's high Arctic), should
the country's "climate" reflect where farming actually happened, not
where empty desert or tundra dominates the area?

We use HYDE 3.5 gridded cropland and grazing land at 5-arcminute
resolution, averaged over the pre-industrial reference window 1500--1700.
Cells with no agricultural land (deserts, tundra) get zero weight; cells
with intensive farming or pasture get high weight. This is more aggressive
than pop-weighting and most direct for the agricultural-pathway question.

For countries with very sparse pre-industrial agriculture, the result is
unstable (concentrated in 1-2 cells). We retain area- and pop-weighted
panels as primary; this is an additional robustness layer.

Outputs:
    analysis/data/modera_country_monthly_cropw.parquet
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"
MODERA = Path("/Volumes/BIGDATA/MODERA/extracted")
HYDE_CROP_NC = ROOT / "gbc2025_7apr_base" / "NetCDF" / "cropland.nc"
HYDE_GRAZ_NC = ROOT / "gbc2025_7apr_base" / "NetCDF" / "grazing_land.nc"
COUNTRY_RASTER = ROOT / "general_files" / "general_files" / "iso_cr.asc"
TEMP_NC = MODERA / "ModE-RA_ensmean_temp2_anom_wrt_1901-2000_1421-2008_mon.nc"
PRECIP_NC = MODERA / "ModE-RA_ensmean_totprec_anom_wrt_1901-2000_1421-2008_mon.nc"

REFERENCE_YEARS = [1500, 1600, 1700]


def _read_country_raster():
    with open(COUNTRY_RASTER) as f:
        h = {}
        for _ in range(6):
            k, v = f.readline().strip().split()
            h[k.lower()] = float(v)
    ncols = int(h["ncols"]); nrows = int(h["nrows"])
    xll = h["xllcorner"]; yll = h["yllcorner"]; cs = h["cellsize"]; nodata = h["nodata_value"]
    data = np.loadtxt(COUNTRY_RASTER, skiprows=6, dtype=np.float64).astype(np.int64)
    data[data == int(nodata)] = -1
    lon = xll + (np.arange(ncols) + 0.5) * cs
    lat = yll + (np.arange(nrows)[::-1] + 0.5) * cs
    return data, lat, lon


def _load_ag_footprint() -> np.ndarray:
    """Mean of (cropland + 0.5 * grazing) over REFERENCE_YEARS at 5-arcmin."""
    print(f"Loading HYDE gridded cropland+0.5*grazing over {REFERENCE_YEARS}...",
          flush=True)
    cds = xr.open_dataset(HYDE_CROP_NC,
                          decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    gds = xr.open_dataset(HYDE_GRAZ_NC,
                          decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    yrs = np.array([t.year for t in cds["time"].values])
    idx = [i for i, y in enumerate(yrs) if y in REFERENCE_YEARS]
    crop = cds["cropland"].isel(time=idx).mean("time").values
    graz = gds["grazing_land"].isel(time=idx).mean("time").values
    ag = np.nan_to_num(crop, nan=0.0) + 0.5 * np.nan_to_num(graz, nan=0.0)
    ag = np.clip(ag, 0, None)
    print(f"  ag grid shape: {ag.shape}, "
          f"total {ag.sum():.0f}, "
          f"non-zero cells: {(ag > 0).sum():,}/{ag.size:,}")
    # Also keep a small fallback weight (cos lat × 1e-6) for countries with
    # zero pre-industrial agriculture — otherwise their climate is undefined.
    lat = cds["lat"].values
    coslat = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, ag.shape[1]))
    ag_with_floor = ag + coslat * 1e-6
    return ag_with_floor


def main() -> None:
    codes, lat_c, lon_c = _read_country_raster()
    ag = _load_ag_footprint()

    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    iso3_list = sorted(set(num_to_iso3.values()))
    iso3_to_idx = {c: i for i, c in enumerate(iso3_list)}

    with xr.open_dataset(TEMP_NC,
                          decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        m_lat = ds["latitude"].values
        m_lon = ds["longitude"].values
    n_lat_t = m_lat.size; n_lon_t = m_lon.size

    print("Building country weight matrix (cropland+grazing-weighted)...", flush=True)
    lat_idx = np.argmin(np.abs(lat_c[:, None] - m_lat[None, :]), axis=1).astype(np.int32)
    lon_idx = np.argmin(np.abs(lon_c[:, None] - m_lon[None, :]), axis=1).astype(np.int32)
    rows, cols, vals = [], [], []
    for i in range(codes.shape[0]):
        cr = codes[i]
        valid = cr > 0
        if not valid.any(): continue
        li = lat_idx[i]
        for j in np.flatnonzero(valid):
            w = float(ag[i, j])
            if w <= 0: continue
            iso3 = num_to_iso3.get(int(cr[j]))
            if iso3 is None: continue
            rows.append(iso3_to_idx[iso3])
            cols.append(li * n_lon_t + lon_idx[j])
            vals.append(w)
        if (i + 1) % 400 == 0:
            print(f"  raster row {i+1}/{codes.shape[0]}", flush=True)
    W = sparse.coo_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(len(iso3_list), n_lat_t * n_lon_t),
    ).tocsr()
    rs = np.asarray(W.sum(axis=1)).ravel()
    print(f"  W nnz {W.nnz:,}, countries with positive weight: "
          f"{(rs > 0).sum()}/{len(iso3_list)}")

    inv = np.where(rs > 0, 1.0 / rs, 0.0)
    print("Loading temp anomalies...", flush=True)
    with xr.open_dataset(TEMP_NC,
                          decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        t = ds["temp2"].values.astype(np.float64)
        time = ds["time"].values
    print("Aggregating temp...", flush=True)
    t_flat = t.reshape(t.shape[0], -1)
    t_out = (W @ t_flat.T) * inv[:, None]

    print("Loading precip anomalies...", flush=True)
    with xr.open_dataset(PRECIP_NC,
                          decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        p = ds["totprec"].values.astype(np.float64)
    print("Aggregating precip...", flush=True)
    p_flat = p.reshape(p.shape[0], -1)
    p_flux = (W @ p_flat.T) * inv[:, None]

    years = np.array([t.year for t in time], dtype=np.int32)
    months = np.array([t.month for t in time], dtype=np.int8)
    days = np.array([pd.Period(year=int(y), month=int(m), freq="M").days_in_month
                       for y, m in zip(years, months)], dtype=np.float64)
    p_mm = p_flux * 86400.0 * days

    n_c, n_t = t_out.shape
    df = pd.DataFrame({
        "iso3": np.repeat(np.asarray(iso3_list), n_t),
        "year": np.tile(years, n_c),
        "month": np.tile(months, n_c),
        "t_anom_c": t_out.ravel().astype(np.float32),
        "p_anom_mm": p_mm.ravel().astype(np.float32),
    })
    keep = pd.DataFrame({"iso3": iso3_list, "agw_sum": rs})
    keep = keep[keep["agw_sum"] > 0]
    df = df[df["iso3"].isin(keep["iso3"])]
    out = DATA / "modera_country_monthly_cropw.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df):,} rows, {df['iso3'].nunique()} countries)")
    keep.to_parquet(DATA / "cropweight_diagnostic.parquet", index=False)


if __name__ == "__main__":
    main()
