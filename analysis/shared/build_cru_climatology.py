"""Aggregate CRU TS 4.09 (1901-1950) to country-month and build a 1901-1950
climatology (country x calendar-month means).

CRU TS native grid is 0.5 deg (720x360). We use the same iso_cr 5-arcmin
raster as ModE-RA aggregation, weighting by cos(lat).

Outputs:
    analysis/data/cru_country_monthly_1901_1950.parquet
    analysis/data/cru_country_climatology_1901_1950.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

ROOT = Path("/Volumes/BIGDATA/HYDE35")
CRU_DIR = ROOT / "climate_reconstructions" / "cru_ts"
COUNTRY_RASTER = ROOT / "general_files" / "general_files" / "iso_cr.asc"
COUNTRY_MAP_CSV = ROOT / "hyde35_country_iso_mapping.csv"
OUT_DIR = ROOT / "analysis" / "data"


def _read_country_raster() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(COUNTRY_RASTER) as f:
        header = {}
        for _ in range(6):
            k, v = f.readline().strip().split()
            header[k.lower()] = float(v)
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    xll = header["xllcorner"]
    yll = header["yllcorner"]
    cs = header["cellsize"]
    nodata = header["nodata_value"]
    data = np.loadtxt(COUNTRY_RASTER, skiprows=6, dtype=np.int32)
    data[data == int(nodata)] = -1
    lon = xll + (np.arange(ncols) + 0.5) * cs
    lat = yll + (np.arange(nrows)[::-1] + 0.5) * cs
    return data, lat, lon


def _build_weight_matrix_to(target_lat: np.ndarray, target_lon: np.ndarray) -> tuple[sparse.csr_matrix, list[str]]:
    codes, lat_c, lon_c = _read_country_raster()
    mp = pd.read_csv(COUNTRY_MAP_CSV)
    mp = mp.dropna(subset=["iso_num", "iso3"]).copy()
    mp["iso_num"] = mp["iso_num"].astype(int)
    iso_num_to_iso3 = dict(zip(mp["iso_num"], mp["iso3"]))
    iso3_list = sorted(mp["iso3"].unique().tolist())
    iso3_to_idx = {c: i for i, c in enumerate(iso3_list)}
    n_lat = target_lat.size
    n_lon = target_lon.size
    lat_idx = np.argmin(np.abs(lat_c[:, None] - target_lat[None, :]), axis=1).astype(np.int32)
    lon_idx = np.argmin(np.abs(lon_c[:, None] - target_lon[None, :]), axis=1).astype(np.int32)
    lat_w = np.cos(np.deg2rad(lat_c))
    rows, cols, weights = [], [], []
    for i in range(codes.shape[0]):
        code_row = codes[i]
        valid = code_row > 0
        if not valid.any():
            continue
        li = lat_idx[i]
        wgt = lat_w[i]
        for j in np.flatnonzero(valid):
            iso_num = int(code_row[j])
            iso3 = iso_num_to_iso3.get(iso_num)
            if iso3 is None:
                continue
            rows.append(iso3_to_idx[iso3])
            cols.append(li * n_lon + lon_idx[j])
            weights.append(wgt)
    W = sparse.coo_matrix(
        (np.asarray(weights, dtype=np.float64),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(len(iso3_list), n_lat * n_lon),
    ).tocsr()
    return W, iso3_list


def _aggregate(W: sparse.csr_matrix, da: xr.DataArray) -> np.ndarray:
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    arr = da.values.astype(np.float64)  # (time, lat, lon)
    arr = np.nan_to_num(arr, nan=0.0)
    n_time = arr.shape[0]
    flat = arr.reshape(n_time, -1)
    out = (W @ flat.T) * inv[:, None]
    return out


def main() -> None:
    decades = [(1901, 1910), (1911, 1920), (1921, 1930), (1931, 1940), (1941, 1950)]
    tmp_paths = [CRU_DIR / f"cru_ts4.09.{a}.{b}.tmp.dat.nc" for a, b in decades]
    pre_paths = [CRU_DIR / f"cru_ts4.09.{a}.{b}.pre.dat.nc" for a, b in decades]

    print("Opening CRU TS tmp files...", flush=True)
    tmp_ds = xr.concat([xr.open_dataset(p) for p in tmp_paths], dim="time")
    pre_ds = xr.concat([xr.open_dataset(p) for p in pre_paths], dim="time")
    print(f"  tmp time: {tmp_ds.sizes['time']} months, "
          f"{tmp_ds['time'].values[0]} .. {tmp_ds['time'].values[-1]}", flush=True)

    cru_lat = tmp_ds["lat"].values.astype(np.float64)
    cru_lon = tmp_ds["lon"].values.astype(np.float64)

    print("Building weight matrix (countries x CRU cells)...", flush=True)
    W, iso3_list = _build_weight_matrix_to(cru_lat, cru_lon)
    print(f"  nnz = {W.nnz:,}", flush=True)

    print("Aggregating tmp...", flush=True)
    tmp = _aggregate(W, tmp_ds["tmp"])
    print("Aggregating pre...", flush=True)
    pre = _aggregate(W, pre_ds["pre"])

    times = pd.DatetimeIndex(tmp_ds["time"].values)
    years = times.year.values.astype(np.int16)
    months = times.month.values.astype(np.int8)
    n_c, n_t = tmp.shape

    monthly = pd.DataFrame({
        "iso3": np.repeat(np.asarray(iso3_list), n_t),
        "year": np.tile(years, n_c),
        "month": np.tile(months, n_c),
        "tmp_c": tmp.ravel().astype(np.float32),
        "pre_mm": pre.ravel().astype(np.float32),
    })
    out_m = OUT_DIR / "cru_country_monthly_1901_1950.parquet"
    monthly.to_parquet(out_m, index=False)
    print(f"Wrote {out_m} ({len(monthly):,} rows)", flush=True)

    # Climatology: country x calendar-month means
    clim = monthly.groupby(["iso3", "month"], as_index=False).agg(
        tmp_c_clim=("tmp_c", "mean"),
        pre_mm_clim=("pre_mm", "mean"),
    )
    out_c = OUT_DIR / "cru_country_climatology_1901_1950.parquet"
    clim.to_parquet(out_c, index=False)
    print(f"Wrote {out_c} ({len(clim):,} rows)", flush=True)


if __name__ == "__main__":
    main()
