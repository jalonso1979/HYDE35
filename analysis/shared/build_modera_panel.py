"""Build a country-monthly climate panel from ModE-RA 1421-2008.

ModE-RA provides monthly temperature and total-precipitation anomalies
(reference 1901-2000) on a global 192x96 grid (1.875 deg lon, Gaussian lat).
We area-weight onto 197 HYDE countries via the 5-arcmin iso_cr.asc raster.

Output: analysis/data/modera_country_monthly.parquet with columns
    iso3, year, month, t_anom_c, p_anom_mm_per_month, area_weight_sum
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

ROOT = Path("/Volumes/BIGDATA/HYDE35")
MODERA_DIR = Path("/Volumes/BIGDATA/MODERA/extracted")
COUNTRY_RASTER = ROOT / "general_files" / "general_files" / "iso_cr.asc"
COUNTRY_MAP_CSV = ROOT / "hyde35_country_iso_mapping.csv"
OUT_DIR = ROOT / "analysis" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_NC = MODERA_DIR / "ModE-RA_ensmean_temp2_anom_wrt_1901-2000_1421-2008_mon.nc"
PRECIP_NC = MODERA_DIR / "ModE-RA_ensmean_totprec_anom_wrt_1901-2000_1421-2008_mon.nc"


def _read_country_raster() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (codes 2D int, lat centers, lon centers) for the iso_cr 5-arcmin grid."""
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


def _build_weight_matrix() -> tuple[sparse.csr_matrix, list[str], pd.DataFrame]:
    """Build sparse (n_country x n_modera_cell) area-weight matrix."""
    print("Loading country raster (4320x2160)...", flush=True)
    codes, lat_c, lon_c = _read_country_raster()
    print(f"  raster shape: {codes.shape}, unique codes: {np.unique(codes).size}", flush=True)

    mp = pd.read_csv(COUNTRY_MAP_CSV)
    mp = mp.dropna(subset=["iso_num", "iso3"]).copy()
    mp["iso_num"] = mp["iso_num"].astype(int)
    iso_num_to_iso3 = dict(zip(mp["iso_num"], mp["iso3"]))
    iso3_list = sorted(mp["iso3"].unique().tolist())
    iso3_to_idx = {c: i for i, c in enumerate(iso3_list)}

    print("Loading ModE-RA grid...", flush=True)
    with xr.open_dataset(TEMP_NC, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        m_lat = ds["latitude"].values
        m_lon = ds["longitude"].values
    n_lat = m_lat.size
    n_lon = m_lon.size
    n_cells = n_lat * n_lon

    print("Mapping country-raster cells -> ModE-RA cells...", flush=True)
    lat_idx = np.argmin(np.abs(lat_c[:, None] - m_lat[None, :]), axis=1).astype(np.int32)
    lon_idx = np.argmin(np.abs(lon_c[:, None] - m_lon[None, :]), axis=1).astype(np.int32)
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
            r = iso3_to_idx[iso3]
            c = li * n_lon + lon_idx[j]
            rows.append(r)
            cols.append(c)
            weights.append(wgt)
        if (i + 1) % 200 == 0:
            print(f"  row {i + 1}/{codes.shape[0]}", flush=True)

    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)

    W = sparse.coo_matrix(
        (weights, (rows, cols)), shape=(len(iso3_list), n_cells)
    ).tocsr()
    # row-sum once for normalization
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    print(f"  weight matrix: nnz={W.nnz:,}, "
          f"countries with positive weight: {(row_sums > 0).sum()}/{len(iso3_list)}",
          flush=True)
    coverage = pd.DataFrame({
        "iso3": iso3_list,
        "weight_sum": row_sums,
        "has_coverage": row_sums > 0,
    })
    return W, iso3_list, coverage


def aggregate_variable(var_nc: Path, var_name: str, scale: float = 1.0) -> np.ndarray:
    """Return (n_country, n_time) array of area-weighted means."""
    W, iso3_list, coverage = _build_weight_matrix()
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)

    print(f"Loading {var_name} from {var_nc.name}...", flush=True)
    with xr.open_dataset(var_nc, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        arr = ds[var_name].values  # (time, lat, lon)
        time = ds["time"].values
    n_time = arr.shape[0]
    arr2 = arr.reshape(n_time, -1).astype(np.float64) * scale  # (time, n_cells)

    # country-mean = (W @ arr2.T) / row_sums per country
    print("Aggregating to countries...", flush=True)
    out = (W @ arr2.T) * inv[:, None]  # (n_country, n_time)
    years = np.array([t.year for t in time], dtype=np.int32)
    months = np.array([t.month for t in time], dtype=np.int8)
    return iso3_list, years, months, out, coverage


def main() -> None:
    iso3_list, years, months, t_anom, coverage = aggregate_variable(
        TEMP_NC, "temp2", scale=1.0
    )
    # K anomaly == C anomaly (delta), keep name explicit
    _, years_p, months_p, p_anom_flux, _ = aggregate_variable(
        PRECIP_NC, "totprec", scale=1.0
    )
    assert np.array_equal(years, years_p) and np.array_equal(months, months_p)

    # precip is in kg m-2 s-1 -> mm/day = * 86400; convert to mm/month (days/month)
    days_per_month = np.array(
        [pd.Period(year=int(y), month=int(m), freq="M").days_in_month
         for y, m in zip(years, months)],
        dtype=np.float64,
    )
    p_anom_mm = p_anom_flux * 86400.0 * days_per_month  # (n_country, n_time)

    print("Building long-format panel...", flush=True)
    n_c, n_t = t_anom.shape
    iso3_arr = np.repeat(np.asarray(iso3_list), n_t)
    year_arr = np.tile(years, n_c)
    month_arr = np.tile(months, n_c)
    df = pd.DataFrame({
        "iso3": iso3_arr,
        "year": year_arr,
        "month": month_arr,
        "t_anom_c": t_anom.ravel().astype(np.float32),
        "p_anom_mm": p_anom_mm.ravel().astype(np.float32),
    })

    has_cov = coverage.set_index("iso3")["has_coverage"]
    df = df[df["iso3"].map(has_cov).fillna(False)].copy()
    out_path = OUT_DIR / "modera_country_monthly.parquet"
    df.to_parquet(out_path, index=False)
    cov_path = OUT_DIR / "modera_country_coverage.parquet"
    coverage.to_parquet(cov_path, index=False)
    print(f"Wrote {out_path} ({len(df):,} rows, "
          f"{df['iso3'].nunique()} countries, {df['year'].min()}-{df['year'].max()})",
          flush=True)
    print(f"Wrote {cov_path}", flush=True)


if __name__ == "__main__":
    main()
