"""Build annual England ModE-RA growing-season + winter climate panel.

Source parquet: data/modera_country_monthly_cropw.parquet  (cropland-weighted
ModE-RA monthly anomalies by country, built by paper4_shadow).

ERA5 splice for 2009+ from data/era5_country_monthly.parquet if present.
ERA5 provides absolute levels (t2m_c, tp_mm); we convert to anomalies relative
to the ERA5 1950-2000 monthly climatology so the splice is unit-consistent
with the ModE-RA anomalies.

Outputs columns:
    year, t_growing, t_winter, p_growing, p_winter, vol_t_10y, vol_p_10y, source
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis")
MODERA = ROOT / "data" / "modera_country_monthly_cropw.parquet"
ERA5 = ROOT / "data" / "era5_country_monthly.parquet"
OUT = ROOT / "data" / "long_shadow_fertility" / "england_climate_annual_1421_2008.parquet"

GROW_MONTHS = [4, 5, 6, 7, 8, 9]
WINTER_MONTHS = [10, 11, 12, 1, 2, 3]
MODERA_START = 1421
MODERA_END = 2008
ERA5_SPLICE_START = 2009


def _modera_england_monthly() -> pd.DataFrame:
    mod = pd.read_parquet(MODERA)
    gbr = mod.loc[mod["iso3"] == "GBR", ["year", "month", "t_anom_c", "p_anom_mm"]].copy()
    gbr["source"] = "ModE-RA_cropw_GBR"
    return gbr.sort_values(["year", "month"]).reset_index(drop=True)


def _era5_england_monthly_anom() -> pd.DataFrame | None:
    """Return ERA5 GBR monthly anomalies (relative to the 1950-2000 climatology)
    for 2009 onward, with a `source` column. Returns None if file is missing or
    has no usable 2009+ rows."""
    if not ERA5.exists():
        return None
    e = pd.read_parquet(ERA5)
    gbr = e.loc[e["iso3"] == "GBR", ["year", "month", "t2m_c", "tp_mm"]].copy()
    if gbr.empty:
        return None
    ref = (
        gbr.loc[gbr["year"].between(1950, 2000)]
        .groupby("month", as_index=False)
        .agg(t_ref=("t2m_c", "mean"), p_ref=("tp_mm", "mean"))
    )
    if ref.empty or len(ref) < 12:
        return None
    gbr = gbr.merge(ref, on="month", how="left")
    gbr["t_anom_c"] = gbr["t2m_c"] - gbr["t_ref"]
    gbr["p_anom_mm"] = gbr["tp_mm"] - gbr["p_ref"]
    gbr = gbr.loc[gbr["year"] >= ERA5_SPLICE_START, ["year", "month", "t_anom_c", "p_anom_mm"]].copy()
    if gbr.empty:
        return None
    gbr["source"] = "ERA5_2009_modern"
    return gbr.sort_values(["year", "month"]).reset_index(drop=True)


def _annual_seasonal_means(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly anomalies to growing-season (Apr-Sep) and winter (Oct-Mar) means.

    Winter year Y covers Oct Y, Nov Y, Dec Y, Jan Y+1, Feb Y+1, Mar Y+1.
    Returns rows only for years with the full 6-month winter season available
    (so partial seasons at the ends of the panel are dropped)."""
    m = monthly.copy()
    m["winter_year"] = np.where(m["month"] >= 10, m["year"], m["year"] - 1)
    grow_mask = m["month"].isin(GROW_MONTHS)
    winter_mask = m["month"].isin(WINTER_MONTHS)
    grow = (
        m.loc[grow_mask]
        .groupby("year", as_index=False)
        .agg(
            t_growing=("t_anom_c", "mean"),
            p_growing=("p_anom_mm", "mean"),
            n_grow=("month", "count"),
        )
    )
    grow = grow.loc[grow["n_grow"] == 6].drop(columns=["n_grow"])
    win = (
        m.loc[winter_mask]
        .groupby("winter_year", as_index=False)
        .agg(
            t_winter=("t_anom_c", "mean"),
            p_winter=("p_anom_mm", "mean"),
            n_win=("month", "count"),
        )
        .rename(columns={"winter_year": "year"})
    )
    win = win.loc[win["n_win"] == 6].drop(columns=["n_win"])
    return grow.merge(win, on="year", how="outer")


def _rolling_volatility(monthly: pd.DataFrame, var: str, win: int = 10) -> pd.DataFrame:
    """Compute the 10-year rolling mean of within-year monthly-anomaly SDs.

    Returns a (year, vol_<var>_<win>y) DataFrame indexed by all distinct years
    in `monthly`. `min_periods=5` allows the rolling window to start filling
    after 5 years."""
    annual = monthly.groupby("year", as_index=False).agg(x=(var, "std"))
    annual = annual.sort_values("year").reset_index(drop=True)
    annual[f"vol_{var}_{win}y"] = annual["x"].rolling(window=win, min_periods=5).mean()
    return annual[["year", f"vol_{var}_{win}y"]]


def build_england_climate_annual(write: bool = False) -> pd.DataFrame:
    """Build the annual England climate panel.

    Coverage:
        - ModE-RA cropland-weighted anomalies for 1421-2008.
        - Optional ERA5 splice (converted to anomalies vs 1950-2000) for 2009+.

    Rolling volatility is computed on the unified monthly anomaly series so the
    splice junction does not produce NaN values."""
    modera_m = _modera_england_monthly()
    era5_m = _era5_england_monthly_anom()
    if era5_m is not None:
        monthly = pd.concat(
            [modera_m, era5_m[["year", "month", "t_anom_c", "p_anom_mm", "source"]]],
            ignore_index=True,
        )
    else:
        monthly = modera_m.copy()
    monthly = monthly.sort_values(["year", "month"]).reset_index(drop=True)

    seas = _annual_seasonal_means(monthly)
    # Anchor to growing-season years only: require Apr-Sep present.
    seas = seas.dropna(subset=["t_growing"]).copy()

    vol_t = _rolling_volatility(monthly, "t_anom_c", 10).rename(
        columns={"vol_t_anom_c_10y": "vol_t_10y"}
    )
    vol_p = _rolling_volatility(monthly, "p_anom_mm", 10).rename(
        columns={"vol_p_anom_mm_10y": "vol_p_10y"}
    )
    df = seas.merge(vol_t, on="year", how="left").merge(vol_p, on="year", how="left")

    # Attach source label by year (first source observed for that year).
    src = (
        monthly.groupby("year", as_index=False)
        .agg(source=("source", "first"))
    )
    df = df.merge(src, on="year", how="left")
    df = df.loc[df["year"].between(MODERA_START, 2100)].sort_values("year").reset_index(drop=True)

    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_england_climate_annual(write=True)
    print(f"wrote {OUT} ({len(df)} rows, {df['year'].min()}-{df['year'].max()})")
    print(df["source"].value_counts(dropna=False).to_string())
