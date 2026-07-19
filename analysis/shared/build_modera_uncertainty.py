"""Build country-monthly ModE-RA ensemble UNCERTAINTY panel using the
ensstd, ensmin, ensmax fields from /Volumes/BIGDATA/MODERA/extracted/.

Output:
    analysis/data/modera_country_uncertainty.parquet
        iso3, year, month, t_std, t_min, t_max, p_std
"""

from __future__ import annotations
from pathlib import Path
import warnings; warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

ROOT = Path("/Volumes/BIGDATA/HYDE35")
MODERA = Path("/Volumes/BIGDATA/MODERA/extracted")
DATA = ROOT / "analysis" / "data"
COUNTRY_RASTER = ROOT / "general_files" / "general_files" / "iso_cr.asc"
COUNTRY_MAP_CSV = ROOT / "hyde35_country_iso_mapping.csv"


def _read_country_raster() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(COUNTRY_RASTER) as f:
        h = {}
        for _ in range(6):
            k, v = f.readline().strip().split()
            h[k.lower()] = float(v)
    ncols = int(h["ncols"]); nrows = int(h["nrows"])
    xll = h["xllcorner"]; yll = h["yllcorner"]
    cs = h["cellsize"]; nodata = h["nodata_value"]
    data = np.loadtxt(COUNTRY_RASTER, skiprows=6, dtype=np.int32)
    data[data == int(nodata)] = -1
    lon = xll + (np.arange(ncols) + 0.5) * cs
    lat = yll + (np.arange(nrows)[::-1] + 0.5) * cs
    return data, lat, lon


def _build_weight_matrix() -> tuple[sparse.csr_matrix, list[str]]:
    codes, lat_c, lon_c = _read_country_raster()
    iso_map = pd.read_csv(COUNTRY_MAP_CSV)
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    iso3_list = sorted(set(num_to_iso3.values()))
    iso3_to_idx = {c: i for i, c in enumerate(iso3_list)}

    with xr.open_dataset(MODERA / "ModE-RA_ensstd_temp2_anom_wrt_1901-2000_1421-2008_mon.nc",
                        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        m_lat = ds["latitude"].values
        m_lon = ds["longitude"].values
    n_lat = m_lat.size; n_lon = m_lon.size

    lat_idx = np.argmin(np.abs(lat_c[:, None] - m_lat[None, :]), axis=1).astype(np.int32)
    lon_idx = np.argmin(np.abs(lon_c[:, None] - m_lon[None, :]), axis=1).astype(np.int32)
    lat_w = np.cos(np.deg2rad(lat_c))

    rows, cols, weights = [], [], []
    for i in range(codes.shape[0]):
        cr = codes[i]
        valid = cr > 0
        if not valid.any():
            continue
        li = lat_idx[i]
        wgt = lat_w[i]
        for j in np.flatnonzero(valid):
            iso3 = num_to_iso3.get(int(cr[j]))
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


def _agg(W: sparse.csr_matrix, path: Path, var: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    with xr.open_dataset(path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        arr = ds[var].values.astype(np.float64)
        time = ds["time"].values
    flat = arr.reshape(arr.shape[0], -1)
    out = (W @ flat.T) * inv[:, None]
    years = np.array([t.year for t in time], dtype=np.int32)
    months = np.array([t.month for t in time], dtype=np.int8)
    return out, years, months


def main() -> None:
    print("Building weight matrix...", flush=True)
    W, iso3_list = _build_weight_matrix()
    print(f"  nnz = {W.nnz:,}", flush=True)

    print("Aggregating ensstd temp...", flush=True)
    t_std, years, months = _agg(W,
        MODERA / "ModE-RA_ensstd_temp2_anom_wrt_1901-2000_1421-2008_mon.nc", "temp2")
    print("Aggregating ensmin temp...", flush=True)
    t_min, _, _ = _agg(W,
        MODERA / "ModE-RA_ensmin_temp2_anom_wrt_1901-2000_1421-2008_mon.nc", "temp2")
    print("Aggregating ensmax temp...", flush=True)
    t_max, _, _ = _agg(W,
        MODERA / "ModE-RA_ensmax_temp2_anom_wrt_1901-2000_1421-2008_mon.nc", "temp2")
    print("Aggregating ensstd precip...", flush=True)
    p_std, _, _ = _agg(W,
        MODERA / "ModE-RA_ensstd_totprec_anom_wrt_1901-2000_1421-2008_mon.nc", "totprec")

    n_c, n_t = t_std.shape
    days = np.array(
        [pd.Period(year=int(y), month=int(m), freq="M").days_in_month
         for y, m in zip(years, months)], dtype=np.float64,
    )
    p_std_mm = p_std * 86400.0 * days  # mm/month

    df = pd.DataFrame({
        "iso3": np.repeat(np.asarray(iso3_list), n_t),
        "year": np.tile(years, n_c),
        "month": np.tile(months, n_c),
        "t_std": t_std.ravel().astype(np.float32),
        "t_min": t_min.ravel().astype(np.float32),
        "t_max": t_max.ravel().astype(np.float32),
        "p_std": p_std_mm.ravel().astype(np.float32),
    })
    out = DATA / "modera_country_uncertainty.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df):,} rows)", flush=True)

    # Summary
    print("\n=== ModE-RA ensemble uncertainty summary ===")
    print(f"Median country-month T-std: {df['t_std'].median():.3f} C")
    print(f"75th pct T-std:             {df['t_std'].quantile(0.75):.3f} C")
    print(f"95th pct T-std:             {df['t_std'].quantile(0.95):.3f} C")
    # By era
    for label, mask in [("1421-1500", df["year"].between(1421, 1500)),
                        ("1500-1700", df["year"].between(1500, 1700)),
                        ("1700-1850", df["year"].between(1700, 1850)),
                        ("1850-2008", df["year"].between(1850, 2008))]:
        sub = df[mask]
        print(f"  {label}: median T-std = {sub['t_std'].median():.3f} C, "
              f"median P-std = {sub['p_std'].median():.2f} mm")


if __name__ == "__main__":
    main()
