"""Build sub-national climate + HYDE panels for within-country analysis.

Sub-national units defined by sub_iso_cr.asc (5-arcmin raster with codes
of the form ISO_num*1000 + sub_id). 3,193 units across 203 countries.

Outputs:
    analysis/data/modera_subnational_monthly.parquet  (1421-2008 monthly)
    analysis/data/subnational_hyde.parquet             (decadal pop, cropland, grazing)
    analysis/data/subnational_features.parquet        (pre-industrial features)
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
SUB_RASTER = ROOT / "general_files" / "general_files" / "sub_iso_cr.asc"
MODERA_DIR = Path("/Volumes/BIGDATA/MODERA/extracted")
TEMP_NC = MODERA_DIR / "ModE-RA_ensmean_temp2_anom_wrt_1901-2000_1421-2008_mon.nc"
PRECIP_NC = MODERA_DIR / "ModE-RA_ensmean_totprec_anom_wrt_1901-2000_1421-2008_mon.nc"


def _read_sub_raster() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    with open(SUB_RASTER) as f:
        h = {}
        for _ in range(6):
            k, v = f.readline().strip().split()
            h[k.lower()] = float(v)
    ncols = int(h["ncols"]); nrows = int(h["nrows"])
    xll = h["xllcorner"]; yll = h["yllcorner"]
    cs = h["cellsize"]; nodata = h["nodata_value"]
    data = np.loadtxt(SUB_RASTER, skiprows=6, dtype=np.float64).astype(np.int64)
    data[data == int(nodata)] = -1
    lon = xll + (np.arange(ncols) + 0.5) * cs
    lat = yll + (np.arange(nrows)[::-1] + 0.5) * cs
    codes = np.unique(data); codes = codes[codes > 0].tolist()
    return data, lat, lon, codes


def _build_weight_matrix() -> tuple[sparse.csr_matrix, list[int]]:
    print("Loading sub_iso_cr raster (4320x2160)...", flush=True)
    codes_grid, lat_c, lon_c, sub_codes = _read_sub_raster()
    sub_to_idx = {c: i for i, c in enumerate(sub_codes)}

    print("Loading ModE-RA grid...", flush=True)
    with xr.open_dataset(TEMP_NC, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        m_lat = ds["latitude"].values
        m_lon = ds["longitude"].values
    n_lat = m_lat.size; n_lon = m_lon.size; n_cells = n_lat * n_lon

    lat_idx = np.argmin(np.abs(lat_c[:, None] - m_lat[None, :]), axis=1).astype(np.int32)
    lon_idx = np.argmin(np.abs(lon_c[:, None] - m_lon[None, :]), axis=1).astype(np.int32)
    lat_w = np.cos(np.deg2rad(lat_c))

    rows, cols, weights = [], [], []
    for i in range(codes_grid.shape[0]):
        cr = codes_grid[i]
        valid = cr > 0
        if not valid.any():
            continue
        li = lat_idx[i]
        wgt = lat_w[i]
        for j in np.flatnonzero(valid):
            sub_id = int(cr[j])
            if sub_id not in sub_to_idx:
                continue
            rows.append(sub_to_idx[sub_id])
            cols.append(li * n_lon + lon_idx[j])
            weights.append(wgt)
        if (i+1) % 400 == 0:
            print(f"  row {i+1}/{codes_grid.shape[0]}", flush=True)

    W = sparse.coo_matrix(
        (np.asarray(weights, dtype=np.float64),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(len(sub_codes), n_cells),
    ).tocsr()
    return W, sub_codes


def aggregate_to_subnational() -> None:
    W, sub_codes = _build_weight_matrix()
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)

    print("Loading temp anomalies...", flush=True)
    with xr.open_dataset(TEMP_NC, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        t = ds["temp2"].values.astype(np.float64)
        time = ds["time"].values
    print("Aggregating temp...", flush=True)
    t_flat = t.reshape(t.shape[0], -1)
    t_out = (W @ t_flat.T) * inv[:, None]   # (sub, time)

    print("Loading precip anomalies...", flush=True)
    with xr.open_dataset(PRECIP_NC, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        p = ds["totprec"].values.astype(np.float64)
    print("Aggregating precip...", flush=True)
    p_flat = p.reshape(p.shape[0], -1)
    p_flux = (W @ p_flat.T) * inv[:, None]  # kg/m^2/s

    years = np.array([t.year for t in time], dtype=np.int32)
    months = np.array([t.month for t in time], dtype=np.int8)
    days_per_month = np.array(
        [pd.Period(year=int(y), month=int(m), freq="M").days_in_month
         for y, m in zip(years, months)], dtype=np.float64,
    )
    p_mm = p_flux * 86400.0 * days_per_month  # mm/month

    n_sub, n_t = t_out.shape
    df = pd.DataFrame({
        "sub_id": np.repeat(np.asarray(sub_codes), n_t),
        "year": np.tile(years, n_sub),
        "month": np.tile(months, n_sub),
        "t_anom_c": t_out.ravel().astype(np.float32),
        "p_anom_mm": p_mm.ravel().astype(np.float32),
    })
    df["iso_num"] = (df["sub_id"] // 1000).astype(np.int32)
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    df["iso3"] = df["iso_num"].map(num_to_iso3)
    df = df[df["iso3"].notna()].copy()

    # Save only sub-units with positive weight
    keep = pd.DataFrame({"sub_id": sub_codes, "weight_sum": row_sums})
    keep = keep[keep["weight_sum"] > 0]
    df = df[df["sub_id"].isin(keep["sub_id"])]

    out = DATA / "modera_subnational_monthly.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df):,} rows, "
          f"{df['sub_id'].nunique()} sub-units, "
          f"{df['iso3'].nunique()} countries)", flush=True)


def build_subnational_hyde() -> None:
    """Wide-format HYDE sub-national data: pop, cropland, grazing."""
    base = ROOT / "gbc2025_7apr_base"
    def load(stem: str) -> pd.DataFrame:
        df = pd.read_csv(base / f"{stem}_4apr2025.csv")
        df = df.dropna(subset=["isolink"]).copy()
        df["sub_id"] = df["isolink"].astype(int)
        df = df.drop(columns=["isolink"])
        # melt
        ycols = [c for c in df.columns if c.startswith("y")]
        long = df.melt(id_vars=["sub_id"], value_vars=ycols,
                       var_name="year_col", value_name=stem)
        long["year"] = long["year_col"].str.lstrip("y").astype(int)
        return long[["sub_id", "year", stem]]

    pop = load("subpop")
    crop = pd.read_csv(base / "sub_hiscrop_4apr2025.csv")
    crop = crop.dropna(subset=["isolink"]).copy()
    crop["sub_id"] = crop["isolink"].astype(int)
    crop = crop.drop(columns=["isolink"])
    ycols = [c for c in crop.columns if c.startswith("y")]
    crop_l = crop.melt(id_vars=["sub_id"], value_vars=ycols,
                       var_name="year_col", value_name="cropland_ha")
    crop_l["year"] = crop_l["year_col"].str.lstrip("y").astype(int)
    crop_l = crop_l[["sub_id", "year", "cropland_ha"]]

    past = pd.read_csv(base / "sub_hispast_4apr2025.csv")
    past = past.dropna(subset=["isolink"]).copy()
    past["sub_id"] = past["isolink"].astype(int)
    past = past.drop(columns=["isolink"])
    ycols = [c for c in past.columns if c.startswith("y")]
    past_l = past.melt(id_vars=["sub_id"], value_vars=ycols,
                       var_name="year_col", value_name="grazing_ha")
    past_l["year"] = past_l["year_col"].str.lstrip("y").astype(int)
    past_l = past_l[["sub_id", "year", "grazing_ha"]]

    panel = pop.merge(crop_l, on=["sub_id", "year"], how="outer")
    panel = panel.merge(past_l, on=["sub_id", "year"], how="outer")
    panel["iso_num"] = (panel["sub_id"] // 1000).astype(int)
    iso_map = pd.read_csv(ROOT / "hyde35_country_iso_mapping.csv")
    iso_map = iso_map.dropna(subset=["iso_num", "iso3"]).copy()
    iso_map["iso_num"] = iso_map["iso_num"].astype(int)
    num_to_iso3 = dict(zip(iso_map["iso_num"], iso_map["iso3"]))
    panel["iso3"] = panel["iso_num"].map(num_to_iso3)
    panel = panel[panel["iso3"].notna()].copy()

    out = DATA / "subnational_hyde.parquet"
    panel.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(panel):,} rows, "
          f"{panel['sub_id'].nunique()} sub-units)", flush=True)


def main() -> None:
    aggregate_to_subnational()
    build_subnational_hyde()


if __name__ == "__main__":
    main()
