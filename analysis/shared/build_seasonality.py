"""Compute country-year intra-annual seasonality and other climate measures
from ModE-RA anomalies + CRU 1901-1950 climatology.

Outputs analysis/data/country_seasonality_1421_2008.parquet with columns:
    iso3, year,
    t_jan ... t_dec       (absolute monthly temperature, deg C)
    p_jan ... p_dec       (absolute monthly precipitation, mm)
    sigma_s               max(T_m) - min(T_m)            -- Matranga measure
    sigma_s_std           std(T_m) across 12 months
    t_mean                annual mean temperature
    p_mean                annual mean monthly precip
    p_annual              total annual precip
    growing_dd            growing degree days base 10C
    frost_months          number of months with mean T < 0
    monsoon_intensity     max monthly precip - min monthly precip
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35")
DATA = ROOT / "analysis" / "data"


def main() -> None:
    print("Loading ModE-RA monthly panel...", flush=True)
    mod = pd.read_parquet(DATA / "modera_country_monthly.parquet")
    clim = pd.read_parquet(DATA / "cru_country_climatology_1901_1950.parquet")

    # Inner-join on (iso3, month) so we drop countries CRU lacks (small island nations).
    mod = mod.merge(clim, on=["iso3", "month"], how="inner")
    mod["t_abs"] = mod["t_anom_c"] + mod["tmp_c_clim"]
    mod["p_abs"] = (mod["p_anom_mm"] + mod["pre_mm_clim"]).clip(lower=0.0)

    print("Pivoting to (iso3, year) panel with 12 monthly cols each...", flush=True)
    pivot = mod.pivot_table(
        index=["iso3", "year"],
        columns="month",
        values=["t_abs", "p_abs"],
        aggfunc="mean",
    )
    pivot.columns = [f"{a}_m{int(b):02d}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    t_cols = [f"t_abs_m{m:02d}" for m in range(1, 13)]
    p_cols = [f"p_abs_m{m:02d}" for m in range(1, 13)]

    print("Computing measures...", flush=True)
    t_arr = pivot[t_cols].to_numpy()
    p_arr = pivot[p_cols].to_numpy()

    out = pd.DataFrame({
        "iso3": pivot["iso3"],
        "year": pivot["year"],
        "sigma_s": np.nanmax(t_arr, axis=1) - np.nanmin(t_arr, axis=1),
        "sigma_s_std": np.nanstd(t_arr, axis=1, ddof=0),
        "t_mean": np.nanmean(t_arr, axis=1),
        "p_mean": np.nanmean(p_arr, axis=1),
        "p_annual": np.nansum(p_arr, axis=1),
        "growing_dd": np.nansum(np.clip(t_arr - 10.0, 0.0, None), axis=1) * 30.0,
        "frost_months": (t_arr < 0).sum(axis=1).astype(np.int8),
        "monsoon_intensity": np.nanmax(p_arr, axis=1) - np.nanmin(p_arr, axis=1),
    })

    # Also keep the t_min / t_max month indices and absolute monthly columns
    out["t_max_month"] = (np.nanargmax(t_arr, axis=1) + 1).astype(np.int8)
    out["t_min_month"] = (np.nanargmin(t_arr, axis=1) + 1).astype(np.int8)

    out_path = DATA / "country_seasonality_1421_2008.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(out):,} rows, "
          f"{out['iso3'].nunique()} countries, "
          f"{out['year'].min()}-{out['year'].max()})", flush=True)

    # Pre-industrial means (per country)
    pre = out[out["year"].between(1421, 1750)]
    long = (
        pre.groupby("iso3", as_index=False)
        .agg(
            sigma_s_preind=("sigma_s", "mean"),
            t_mean_preind=("t_mean", "mean"),
            p_annual_preind=("p_annual", "mean"),
            growing_dd_preind=("growing_dd", "mean"),
            monsoon_preind=("monsoon_intensity", "mean"),
            sigma_v_preind=("t_mean", "std"),  # inter-annual T volatility
        )
    )
    long_path = DATA / "country_seasonality_preindustrial.parquet"
    long.to_parquet(long_path, index=False)
    print(f"Wrote {long_path} ({len(long):,} countries, pre-industrial means)", flush=True)


if __name__ == "__main__":
    main()
