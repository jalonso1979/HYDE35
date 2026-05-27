"""4-country growing-season + winter ModE-RA climate panel.

Generalization of Phase 1 build_england_climate_annual.py iterated over
{GBR, FRA, ITA, SWE}.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.paper4_shadow.long_shadow_fertility.data.within_season_variance import add_within_season_sd
from analysis.paper4_shadow.long_shadow_fertility.data.modera_ensstd import aggregate_ensstd_to_country_year

ROOT = Path("/Volumes/BIGDATA/HYDE35/analysis")
MODERA = ROOT / "data" / "modera_country_monthly_cropw.parquet"
MODERA_ENSSTD = ROOT / "data" / "modera_country_uncertainty.parquet"
OUT = ROOT / "data" / "long_shadow_fertility" / "country_climate_annual.parquet"

COUNTRIES = ["GBR", "FRA", "ITA", "SWE", "BEL", "NLD", "ESP"]


def _annual_seasonal_means(monthly: pd.DataFrame) -> pd.DataFrame:
    grow_mask = monthly["month"].isin([4, 5, 6, 7, 8, 9])
    winter_mask = monthly["month"].isin([10, 11, 12, 1, 2, 3])
    monthly = monthly.copy()
    monthly["winter_year"] = np.where(monthly["month"] >= 10, monthly["year"], monthly["year"] - 1)
    grow = (monthly.loc[grow_mask]
            .groupby(["iso3", "year"], as_index=False)
            .agg(t_growing=("t_anom_c", "mean"),
                 p_growing=("p_anom_mm", "mean"),
                 n_grow=("month", "count")))
    win = (monthly.loc[winter_mask]
           .groupby(["iso3", "winter_year"], as_index=False)
           .agg(t_winter=("t_anom_c", "mean"),
                p_winter=("p_anom_mm", "mean"),
                n_win=("month", "count"))
           .rename(columns={"winter_year": "year"}))
    grow = grow.loc[grow["n_grow"] == 6].drop(columns=["n_grow"])
    win = win.loc[win["n_win"] == 6].drop(columns=["n_win"])
    return grow.merge(win, on=["iso3", "year"], how="outer")


def _rolling_volatility(monthly: pd.DataFrame, var: str, win: int = 10) -> pd.DataFrame:
    annual = (monthly.groupby(["iso3", "year"], as_index=False)
              .agg(x=(var, "std")))
    annual = annual.sort_values(["iso3", "year"])
    col_name = f"vol_{var[0]}_{win}y"  # vol_t_10y or vol_p_10y
    annual[col_name] = (annual.groupby("iso3")["x"]
                         .rolling(window=win, min_periods=5)
                         .mean().reset_index(level=0, drop=True))
    return annual[["iso3", "year", col_name]]


def build_country_climate_annual(write: bool = False) -> pd.DataFrame:
    mod = pd.read_parquet(MODERA)
    sub = mod.loc[mod["iso3"].isin(COUNTRIES),
                   ["iso3", "year", "month", "t_anom_c", "p_anom_mm"]].copy()
    seas = _annual_seasonal_means(sub)
    vol_t = _rolling_volatility(sub, "t_anom_c", 10)
    vol_p = _rolling_volatility(sub, "p_anom_mm", 10)
    df = seas.merge(vol_t, on=["iso3", "year"], how="left").merge(vol_p, on=["iso3", "year"], how="left")

    # Phase 10 Pillar C: uncertainty channel columns ---------------------------
    # 1. within-season realized SD (headline)
    t_sd = add_within_season_sd(sub, var="t_anom_c", season_months=(4, 9))
    p_sd = add_within_season_sd(sub, var="p_anom_mm", season_months=(4, 9))

    # 2. ModE-RA ensstd (comparator) — different parquet, filter to same countries
    ens_monthly = pd.read_parquet(MODERA_ENSSTD)
    ens_monthly = ens_monthly.loc[ens_monthly["iso3"].isin(COUNTRIES)].copy()
    ens_monthly = ens_monthly.rename(columns={"t_std": "ensstd_t", "p_std": "ensstd_p"})
    ens_annual = aggregate_ensstd_to_country_year(ens_monthly, season_months=(4, 9))

    # 3. Merge into annual output
    df = (df
          .merge(t_sd, on=["iso3", "year"], how="left")
          .merge(p_sd, on=["iso3", "year"], how="left")
          .merge(ens_annual, on=["iso3", "year"], how="left"))

    df["source"] = "ModE-RA_cropw"
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    df = build_country_climate_annual(write=True)
    print(f"wrote {OUT}: {len(df)} rows, {df['iso3'].nunique()} countries")
